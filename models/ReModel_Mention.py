import torch
import torch.nn as nn

from .utils import MIN, group_biLinear, process_long_input


class ReModel_Mention(nn.Module):
    def __init__(self, config, bert, cls_token_id, sep_token_id):
        super(ReModel_Mention, self).__init__()
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

    # @staticmethod
    # def get_ht(context, attention, mention_map, entity_map, hts, ht_mask):
    #     batch_size = context.shape[0]
    #     ht_num = hts.sum(-1).bool().sum(-1)
    #     mention_num = entity_map.sum(-1)
    #     mention_att = mention_map @ attention.mean(dim=1) @ mention_map.permute(0, 2, 1)
    #     mention = mention_map @ context
    #     entity = entity_map @ mention
    #
    #     h, t = [], []
    #     for b in range(batch_size):
    #         cur_h, cur_t = [], []
    #         for ht in range(ht_num[b]):
    #             if mention_num[b, hts[b, ht, 0]] == 1 and mention_num[b, hts[b, ht, 1]] == 1:
    #                 cur_h.append(entity[b, hts[b, ht, 0]])
    #                 cur_t.append(entity[b, hts[b, ht, 1]])
    #             else:
    #                 h_index = entity_map[b, hts[b, ht, 0]].nonzero().squeeze(-1)
    #                 t_index = entity_map[b, hts[b, ht, 1]].nonzero().squeeze(-1)
    #                 cur_att = mention_att[b][h_index][:, t_index]
    #                 h_t = cur_att.sum(1) / (cur_att.sum() + MIN)
    #                 t_h = cur_att.sum(0) / (cur_att.sum() + MIN)
    #                 h_mention = h_t @ mention[b][h_index]
    #                 t_mention = t_h @ mention[b][t_index]
    #                 cur_h.append(h_mention)
    #                 cur_t.append(t_mention)
    #         h.append(F.pad(torch.stack(cur_h, dim=0), pad=(0, 0, 0, max(ht_num) - ht_num[b])))
    #         t.append(F.pad(torch.stack(cur_t, dim=0), pad=(0, 0, 0, max(ht_num) - ht_num[b])))
    #
    #     return torch.stack(h, dim=0), torch.stack(t, dim=0)
    @staticmethod
    def get_ht(context, attention, mention_map, entity_map, hts, ht_mask):
        batch_size = context.shape[0]
        mention = mention_map @ context
        mention_att = mention_map @ attention.mean(dim=1) @ mention_map.permute(0, 2, 1)
        entity_att = entity_map @ mention_att
        h_mask = torch.stack([entity_map[i, hts[i, :, 0]] for i in range(batch_size)]) * ht_mask
        t_mask = torch.stack([entity_map[i, hts[i, :, 1]] for i in range(batch_size)]) * ht_mask
        h_att = torch.stack([entity_att[i, hts[i, :, 0]] for i in range(batch_size)]) * t_mask
        t_att = torch.stack([entity_att[i, hts[i, :, 1]] for i in range(batch_size)]) * h_mask

        h_att = h_att / (torch.sum(h_att, dim=-1, keepdim=True) + MIN)
        t_att = t_att / (torch.sum(t_att, dim=-1, keepdim=True) + MIN)
        h = (t_att @ mention) * ht_mask
        t = (h_att @ mention) * ht_mask

        return h, t

    @staticmethod
    def context_pooling(context, attention, entity_map, hts, ht_mask):
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

        context_info = context_attention @ context
        return context_info * ht_mask

    def forward(self, **kwargs):
        input_id = kwargs['input_id']
        input_mask = kwargs['input_mask']
        hts = kwargs['hts']
        mention_map = kwargs['mention_map']
        entity_map = kwargs['entity_map']

        ht_mask = (hts.sum(-1) != 0).unsqueeze(-1)

        context, attention = process_long_input(self.bert, input_id, input_mask, self.cls_token_id, self.sep_token_id)
        h, t = self.get_ht(context, attention, mention_map, entity_map, hts, ht_mask)

        entity_map = entity_map @ mention_map
        context_info = self.context_pooling(context, attention, entity_map, hts, ht_mask)

        h = torch.tanh(self.h_dense(h) + self.hc_dense(context_info))
        t = torch.tanh(self.t_dense(t) + self.tc_dense(context_info))
        res = self.clas(h, t)
        # res = res - (~kwargs['type_mask']).float() * MAX

        return res
