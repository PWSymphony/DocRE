import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .utils import MAX, MIN, group_biLinear, process_long_input


class Temp(nn.Module):
    def __init__(self, config, bert, cls_token_id, sep_token_id):
        super(Temp, self).__init__()
        self.bert = bert.requires_grad_(bool(config.pre_lr))
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        bert_hidden_size = self.bert.config.hidden_size
        block_size = 64

        self.h_dense = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.t_dense = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.hc_dense = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.tc_dense = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.he_dense = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.te_dense = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.clas = group_biLinear(bert_hidden_size, config.relation_num, block_size)

    @staticmethod
    def get_ht(context, mention_map, entity_map, hts, ht_mask):
        batch_size = context.shape[0]

        entity_mask = torch.sum(entity_map, dim=-1, keepdim=True) == 0
        mention = mention_map @ context
        mention = torch.exp(mention)
        entity = entity_map @ mention
        entity = torch.masked_fill(entity, entity_mask, 1)
        entity = torch.log(entity)

        h = torch.stack([entity[i, hts[i, :, 0]] for i in range(batch_size)]) * ht_mask
        t = torch.stack([entity[i, hts[i, :, 1]] for i in range(batch_size)]) * ht_mask

        return h, t

    @staticmethod
    def context_pooling(context, attention, entity_map, hts, ht_mask, sent_map, sent_mask, evidences):
        batch_size, max_len, _ = context.shape
        heads = attention.shape[1]

        max_entity = entity_map.shape[1]
        _hts = hts.clone()
        for i in range(batch_size):
            _hts[i] = _hts[i] + i * max_entity
        _hts = _hts.reshape(-1, 2)

        entity_map = entity_map / (torch.sum(entity_map, dim=-1, keepdim=True) + MIN)
        entity_attention = (entity_map.unsqueeze(1) @ attention).permute(1, 0, 2, 3)

        entity_attention = entity_attention.reshape(heads, -1, max_len)
        h_attention = entity_attention[:, _hts[..., 0]].reshape(heads, batch_size, -1, max_len)
        t_attention = entity_attention[:, _hts[..., 1]].reshape(heads, batch_size, -1, max_len)
        context_attention = torch.sum(h_attention * t_attention, dim=0)
        context_attention = context_attention / (torch.sum(context_attention, dim=-1, keepdim=True) + MIN)
        sent_map_avr = sent_map / (torch.sum(sent_map, dim=-1, keepdim=True) + MIN)
        sent_attention = context_attention @ sent_map_avr.permute(0, 2, 1)

        top_index = F.one_hot(sent_attention.topk(3)[1], sent_map_avr.shape[-2]).float().sum(-2)
        evi_info = (top_index @ sent_map) * context_attention
        evi_info = evi_info / (torch.sum(evi_info, dim=-1, keepdim=True) + MIN)
        evi_info = evi_info @ context

        context_info = context_attention @ context
        return context_info, evi_info * ht_mask, sent_attention

    def forward(self, **kwargs):
        input_id = kwargs['input_id']
        input_mask = kwargs['input_mask']
        hts = kwargs['hts']
        mention_map = kwargs['mention_map']
        entity_map = kwargs['entity_map']
        sent_map = kwargs['sent_map']
        sent_mask = kwargs['sent_mask']
        evidences = kwargs['evidences']

        ht_mask = (hts.sum(-1) != 0).unsqueeze(-1)

        context, attention = process_long_input(self.bert, input_id, input_mask, self.cls_token_id, self.sep_token_id)
        h, t = self.get_ht(context, mention_map, entity_map, hts, ht_mask)

        entity_map = entity_map @ mention_map
        context_info, evi_info, sent_attention = self.context_pooling(context, attention, entity_map, hts, ht_mask,
                                                                      sent_map, sent_mask, evidences)
        evi_loss = self.evi_loss(sent_attention, kwargs)

        h = torch.tanh(self.h_dense(h) + self.hc_dense(context_info) + self.he_dense(evi_info))
        t = torch.tanh(self.t_dense(t) + self.tc_dense(context_info) + self.te_dense(evi_info))
        res = self.clas(h, t)

        return res, evi_loss

    @staticmethod
    def evi_loss(pred, batch):
        labels = batch['evidences']
        label_mask = batch['relation_mask']

        have_relation_num = label_mask.sum(-1)
        new_pred = [pred[i, :index] for i, index in enumerate(have_relation_num)]
        labels = [labels[i, :index] for i, index in enumerate(have_relation_num)]
        new_pred = torch.cat(new_pred, dim=0)
        labels = torch.cat(labels, dim=0)
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = new_pred - (1 - p_mask) * MAX
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(-1)
        # loss1 = loss1[type_mask]

        # Rank TH to negative classes
        logit2 = new_pred - (1 - n_mask) * MAX
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(-1)
        # loss2 = loss2[type_mask]

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    @staticmethod
    def get_label(logits, label_mask, num_labels=-1):  # num_labels 是最大的标签数量
        have_relation_num = label_mask.sum(-1).cpu().detach().tolist()
        logits = [logits[i, :index] for i, index in enumerate(have_relation_num)]
        logits = torch.cat(logits, dim=0)

        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)

        output = torch.split(output, have_relation_num, dim=0)
        output = pad_sequence(output, batch_first=True)
        return output
