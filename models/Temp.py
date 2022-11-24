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

        context_attention = torch.exp(context_attention)
        sent_attention = context_attention @ sent_map_avr.permute(0, 2, 1)
        sent_attention = torch.log(sent_attention)

        return sent_attention

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
        sent_attention = self.context_pooling(context, attention, entity_map, hts, ht_mask,
                                              sent_map, sent_mask, evidences)
        sent_attention = torch.masked_fill(sent_attention, ~(sent_mask.unsqueeze(1)), -MAX)
        evi_loss, recall = self.evi_loss(sent_attention, kwargs)

        return recall, evi_loss

    def evi_loss(self, pred, batch):
        labels = batch['evidences']
        hts = batch['hts']
        have_relation_num = (hts.sum(-1) != 0).sum(-1)

        new_pred = [pred[i, :index] for i, index in enumerate(have_relation_num)]
        labels = [labels[i, :index] for i, index in enumerate(have_relation_num)]
        new_pred = torch.cat(new_pred, dim=0)
        labels = torch.cat(labels, dim=0)

        loss = -(F.log_softmax(new_pred, dim=-1) * labels)
        loss = loss.sum(-1).mean()
        recall = self.computer_result(new_pred, labels)

        return loss, recall

    @staticmethod
    def computer_result(pred, label):
        top_index = torch.topk(pred, k=3, dim=-1)[1]
        pred_true_num = 0
        pred_true_num += label.gather(-1, top_index).sum()
        true_num = label.sum()
        return float(pred_true_num), float(true_num)
