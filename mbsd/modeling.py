# coding=utf-8


import math
import os
from typing import List, Optional, Tuple
import collections.abc
import torch
import torch.utils.checkpoint
import torch.nn as nn

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import (
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    _TOKENIZER_FOR_DOC,
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING,
    BertEmbeddings,
    BertEncoder,
    BertPooler,
    BertLMPredictionHead,
)
from .config import ComBertConfig
logger = logging.get_logger(__name__)

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)

class ConvLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

def build_conv_p32(num_channels, embed_dim):
    projection = nn.Sequential(
        nn.Conv2d(num_channels, 64, kernel_size=4, stride=4),
        ConvLayerNorm(64),
        nn.ReLU(),
        nn.Conv2d(64, 192, kernel_size=2, stride=2),
        ConvLayerNorm(192),
        nn.ReLU(),
        nn.Conv2d(192, 384, kernel_size=2, stride=2),
        ConvLayerNorm(384),
        nn.ReLU(),
        nn.Conv2d(384, embed_dim, kernel_size=2, stride=2),
    )
    return projection

def build_conv_p16(num_channels, embed_dim):
    projection = nn.Sequential(
        nn.Conv2d(num_channels, 64, kernel_size=4, stride=4),
        ConvLayerNorm(64),
        nn.ReLU(),
        nn.Conv2d(64, 192, kernel_size=2, stride=2),
        ConvLayerNorm(192),
        nn.ReLU(),
        nn.Conv2d(192, 384, kernel_size=2, stride=2),
        ConvLayerNorm(384),
        nn.ReLU(),
        nn.Conv2d(384, embed_dim, kernel_size=1, stride=1),
    )
    return projection

class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(self, image_size=224, patch_size=16, num_channels=3, embed_dim=768):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # Linear patch embedding
        # self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Convolution patch embedding
        if patch_size[0] == 32:
            self.projection = build_conv_p32(num_channels, embed_dim)
        else:
            self.projection = build_conv_p16(num_channels, embed_dim)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x

class ImageEmbeddings(nn.Module):
    def __init__(self, config, use_mask_token: bool = True):
        super().__init__()
        # patch embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = PatchEmbeddings(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        npatch = embeddings.shape[1] - 1
        N = self.position_embeddings.shape[1] - 1
        if npatch == N and height == width:
            return self.position_embeddings
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode="bicubic",
            align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def mask_embeddings(self, embeddings, bool_masked_pos):
        """
        0.8 replace with [MASK], 0.1 keep and 0.1 replace with other patches.
        Args:
            embeddings: shape=(bsz, num_patches, d_model)
            bool_masked_pos: shape=(bsz, num_patches)
        """
        def binary_mask_like(tensor, prob):
            return torch.bernoulli(torch.full(tensor.shape, prob, device=tensor.device)).bool()
        
        bsz, n_patch, d_model = embeddings.shape
        device = embeddings.device
        replace_mask = bool_masked_pos & binary_mask_like(bool_masked_pos, 0.8)
        rand_mask = bool_masked_pos & (~replace_mask) & binary_mask_like(bool_masked_pos, 0.5)
        random_patches = torch.randint(0, n_patch, (bsz, n_patch), device=device)
        randomized_input = embeddings[torch.arange(bsz).unsqueeze(-1), random_patches]
        embeddings[replace_mask] = self.mask_token.type_as(embeddings)
        embeddings[rand_mask] = randomized_input[rand_mask]

    def forward(self, pixel_values, bool_masked_pos, interpolate_pos_encoding=False):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)
        if bool_masked_pos is not None:
            self.mask_embeddings(embeddings, bool_masked_pos)

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings
        
        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        embeddings = self.dropout(embeddings)

        return embeddings

class MBSDPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ComBertConfig
    load_tf_weights = None
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class MBSDModel(MBSDPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True, use_mask_token=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.img_embeddings = ImageEmbeddings(config, use_mask_token=use_mask_token)
        self.modality_embeddings = nn.Parameter(torch.randn(2, config.hidden_size) * config.initializer_range)

        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        pixel_values=None,
        patch_attention_mask=None,
        bool_masked_pos=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        do_pool=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is None and pixel_values is None:
            ValueError("You have to specify either input_ids or pixel_values")
        if input_ids is not None:
            input_shape = input_ids.shape
            bsz, seq_length = input_shape
            txt_embeddings = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
            )
            txt_embeddings = txt_embeddings + self.modality_embeddings[0]
            if attention_mask is None:
                txt_att_mask = torch.ones(input_shape, device=input_ids.device)
            else:
                txt_att_mask = attention_mask
        if pixel_values is not None:
            img_embeddings = self.img_embeddings(
                pixel_values=pixel_values, 
                bool_masked_pos=bool_masked_pos,
            )
            img_embeddings = img_embeddings + self.modality_embeddings[1]
            input_shape = img_embeddings.shape[:2]
            bsz, img_seq_len = input_shape
            if patch_attention_mask is None:
                img_att_mask = torch.ones(bsz, img_seq_len, device=img_embeddings.device)
            else:
                img_att_mask = torch.cat([patch_attention_mask[:, :1], patch_attention_mask], dim=1)
        if input_ids is None:
            embedding_output = img_embeddings
            attention_mask = img_att_mask
        elif pixel_values is None:
            embedding_output = txt_embeddings
            attention_mask = txt_att_mask
        else:
            input_shape = (bsz, seq_length + img_seq_len)
            embedding_output = torch.cat([txt_embeddings, img_embeddings], dim=1)
            attention_mask = torch.cat([txt_att_mask, img_att_mask], dim=1)
        
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, embedding_output.device)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None and (do_pool is None or do_pool == True) else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class MBSDPretrainHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.img_decoder = nn.Sequential(
            nn.Conv2d(in_channels=config.hidden_size, out_channels=config.patch_size**2 * 3, kernel_size=1),
            nn.PixelShuffle(config.patch_size),
        )

    def forward(self, text_seq_output=None, image_seq_output=None):
        mlm_logits, mpm_logits = None, None
        if text_seq_output is not None:
            mlm_logits = self.predictions(text_seq_output)
        if image_seq_output is not None:
            mpm_logits = self.img_decoder(image_seq_output)
        return mlm_logits, mpm_logits

