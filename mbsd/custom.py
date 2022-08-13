import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modeling import MBSDModel, MBSDPreTrainedModel, MBSDPretrainHead


def masked_patch_pred_loss(mpm_logits, pixel_values, bool_masked_pos):
    """
    Args:
        mpm_logits: shape=(bsz, channel, height, width)
        pixel_values: shape=(bsz, channel, height, width)
    """
    bsz, num_channels, height, width = mpm_logits.shape
    _, num_patches = bool_masked_pos.shape
    p_size = int(math.sqrt(height * width / num_patches))
    size = height // p_size
    bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
    mask = (
        bool_masked_pos.repeat_interleave(p_size, 1)
        .repeat_interleave(p_size, 2)
        .unsqueeze(1)
        .contiguous()
    )
    mpm_loss = F.l1_loss(pixel_values, mpm_logits, reduction="none")
    mpm_loss = (mpm_loss * mask).sum() / (mask.sum() + 1e-5) / num_channels
    return mpm_loss


def binary_kl_div_loss_with_logits(x, y, reduction="batchmean"):
    """This loss combines a sigmoid layer and the Kl_div in one single class.
    """
    logsigmoid_pos = F.logsigmoid(x)
    logsigmoid_neg = logsigmoid_pos - x
    loss_pos = F.kl_div(logsigmoid_pos, y, reduction=reduction)
    loss_neg = F.kl_div(logsigmoid_neg, 1 - y, reduction=reduction)
    loss = loss_pos + loss_neg
    return loss


class MBSDForClassification(MBSDPreTrainedModel):
    def __init__(self, config, num_labels):
        super().__init__(config)

        self.bert = MBSDModel(config, add_pooling_layer=True, use_mask_token=False)
        # Classifier head
        self.proj = nn.Linear(config.hidden_size, num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def encode(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        patch_attention_mask=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
        )
        pooler_output = outputs[1]
        return pooler_output

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        patch_attention_mask=None,
        labels=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
        )
        pooler_output = outputs[1]

        logits = self.proj(pooler_output)

        loss = None
        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logits, labels)
        return loss, logits


class MBSDForItemAlignment(MBSDPreTrainedModel):
    def __init__(self, config, feat_dim):
        super().__init__(config)

        self.bert = MBSDModel(config, add_pooling_layer=True, use_mask_token=False)
        self.proj = nn.Linear(config.hidden_size * 2, 1)

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        src_input_ids=None,
        src_attention_mask=None,
        src_pixel_values=None,
        src_patch_attention_mask=None,
        tgt_input_ids=None,
        tgt_attention_mask=None,
        tgt_pixel_values=None,
        tgt_patch_attention_mask=None,
        labels=None,
        **kwargs
    ):
        outputs = self.bert(
            src_input_ids,
            attention_mask=src_attention_mask,
            pixel_values=src_pixel_values,
            patch_attention_mask=src_patch_attention_mask,
        )
        src = outputs[1]
        outputs = self.bert(
            tgt_input_ids,
            attention_mask=tgt_attention_mask,
            pixel_values=tgt_pixel_values,
            patch_attention_mask=tgt_patch_attention_mask,
        )
        tgt = outputs[1]
        logits = self.proj(torch.cat([src, tgt], dim=1)).squeeze(-1)
        loss = None
        if labels is not None:
            loss_func = nn.BCEWithLogitsLoss()
            loss = loss_func(logits, labels)
        return loss, logits


class MBSDForMatch(MBSDPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = MBSDModel(config, add_pooling_layer=True, use_mask_token=False)
        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        patch_attention_mask=None,
        labels=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
        )
        pooler_output = outputs[1]

        logits = self.classifier(pooler_output).squeeze(-1)

        loss = None
        if labels is not None:
            loss_func = nn.BCEWithLogitsLoss()
            loss = loss_func(logits, labels)
        return loss, logits

class MBSDForPretrainStageI(MBSDPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = MBSDModel(config, add_pooling_layer=False, use_mask_token=True)
        self.cls = MBSDPretrainHead(config)
        self.pixel = config.num_channels * config.patch_size ** 2

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
    
    def reshape_image(self, img_sequence):
        batch_size, sequence_length, num_channels = img_sequence.shape
        height = width = int(sequence_length**0.5)
        img_features = img_sequence.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
        return img_features.contiguous()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        patch_attention_mask=None,
        bool_masked_pos=None,
        labels=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
            bool_masked_pos=bool_masked_pos,
        )
        sequence_output = outputs[0]

        txt_seq_len = input_ids.shape[1]
        img_seq_len = bool_masked_pos.shape[1]
        txt_sequence = sequence_output[:, :txt_seq_len]
        img_sequence = sequence_output[:, -img_seq_len:]
        img_features = self.reshape_image(img_sequence)

        mlm_logits, mpm_logits = self.cls(txt_sequence, img_features)
        
        loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
        mlm_loss = loss_fct(mlm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        mpm_loss = masked_patch_pred_loss(mpm_logits, pixel_values, bool_masked_pos)
        
        loss = mlm_loss + mpm_loss
        return loss, (mlm_loss, mpm_loss), (mlm_logits, mpm_logits)


class MBSDForPretrainStageII(MBSDPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = MBSDModel(config, add_pooling_layer=True, use_mask_token=False)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()
    
    def safe_mean(self, tensor, coef=1):
        return tensor.sum() / (coef * tensor.numel() + 1e-6)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        labels=None,
        **kwargs
    ):
        hidden_q_item = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
        )[1]

        hidden_q_txt = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[1]
        
        t = (attention_mask - token_type_ids).sum(dim=-1).max().item()
        input_ids = input_ids[:, :t]
        attention_mask = attention_mask[:, :t] * (1 - token_type_ids[:, :t])
        hidden_q_img = self.bert(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )[1]
        bsz, hidden_size = hidden_q_item.shape
        logits = self.classifier(hidden_q_item).squeeze(-1)
        logits_q_txt = self.classifier(hidden_q_txt).squeeze(-1)
        logits_q_img = self.classifier(hidden_q_img).squeeze(-1)

        loss_fct = nn.BCEWithLogitsLoss()
        loss_q_item = loss_fct(logits, labels)
        loss_q_txt = loss_fct(logits_q_txt, labels)
        loss_q_img = loss_fct(logits_q_img, labels)

        p_qt = logits_q_txt.sigmoid().detach()
        p_qi = logits_q_img.sigmoid().detach()

        sub_ti = p_qt - p_qi
        sub_it = -sub_ti
        diff_t = labels - p_qt
        diff_i = labels - p_qi
        weight = sub_ti.abs()
        flags_ti = (torch.abs(diff_t) < 0.5) & ((diff_t) * sub_ti > 0)
        flags_it = (torch.abs(diff_i) < 0.5) & ((diff_i) * sub_it > 0)

        with torch.cuda.amp.autocast(enabled=False):
            kld_ti_raw = binary_kl_div_loss_with_logits(logits_q_img.float(), p_qt.float(), reduction="none")
            kld_it_raw = binary_kl_div_loss_with_logits(logits_q_txt.float(), p_qi.float(), reduction="none")
            weight_fp32 = weight.float()
            kld_ti = self.safe_mean((kld_ti_raw * weight_fp32)[flags_ti])
            kld_it = self.safe_mean((kld_it_raw * weight_fp32)[flags_it])

            loss_kld = kld_ti + kld_it

        loss_fct = nn.MSELoss(reduction="none")
        mse_ti_raw = loss_fct(hidden_q_img, hidden_q_txt).sum(dim=1)
        mse_it_raw = loss_fct(hidden_q_txt, hidden_q_img).sum(dim=1)
        mse_ti = self.safe_mean((mse_ti_raw * weight)[flags_ti], hidden_size)
        mse_it = self.safe_mean((mse_it_raw * weight)[flags_it], hidden_size)
        loss_mse = mse_ti + mse_it

        loss = loss_q_item + loss_q_txt + loss_q_img + loss_kld + loss_mse

        return loss, (loss_q_item, loss_q_txt, loss_q_img, loss_kld, loss_mse, kld_ti, kld_it, mse_ti, mse_it)
