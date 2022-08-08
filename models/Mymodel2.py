"""
-*- coding: utf-8 -*-
Time    : 2022/7/24 18:35
Author  : Wang
"""

import torch
import torch.nn as nn
from .long_BERT import process_long_input


class my_model2(nn.Module):
    def __init__(self, config, PTM):
        # 加入关系类别信息，与头尾实体一起context pooling
        super(my_model2, self).__init__()
        self.PTM = PTM.requires_grad_(bool(config.pre_lr))
        PTM_hidden_size = self.PTM.config.hidden_size
        hidden_size = config.hidden_size
        block_size = 64

        self.relation_emb = nn.Embedding(config.relation_num, PTM_hidden_size)
        self.h_dense = nn.Linear(PTM_hidden_size, PTM_hidden_size)
        self.t_dense = nn.Linear(PTM_hidden_size, PTM_hidden_size)
        self.hc_dense = nn.Linear(PTM_hidden_size, PTM_hidden_size)
        self.tc_dense = nn.Linear(PTM_hidden_size, PTM_hidden_size)

        self.clas = nn.Bilinear(PTM_hidden_size, PTM_hidden_size, 1)

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
    def attention(src1, src2, mask, heads=12):
        B, _, H = src1.shape
        assert src2.shape[-1] == H and H % heads == 0
        head_dim = H // heads
        S = head_dim ** (- 0.5)
        src1 = src1.reshape(B, -1, heads, head_dim).permute(0, 2, 1, 3)
        src2 = src2.reshape(B, -1, heads, head_dim).permute(0, 2, 1, 3)
        attn = src1 @ src2.transpose(2, 3) * S
        attn = attn.masked_fill(~mask.reshape(mask.shape[0], 1, 1, -1).bool(), -65504)
        attn = attn.softmax(dim=-1).mean(dim=1)
        return attn

    def context_pooling(self, context, attention, entity_map, hts, ht_mask, relation, input_mask):
        batch_size, max_len, _ = context.shape
        heads = attention.shape[1]

        max_entity = entity_map.shape[1]
        _hts = hts.clone()
        for i in range(batch_size):
            _hts[i] = _hts[i] + i * max_entity
        _hts = _hts.reshape(-1, 2)

        entity_map = entity_map / (torch.sum(entity_map, dim=-1, keepdim=True) + 5.96e-8)
        entity_attention = (entity_map.unsqueeze(1) @ attention).permute(1, 0, 2, 3)

        entity_attention = entity_attention.reshape(heads, -1, max_len)
        h_attention = entity_attention[:, _hts[..., 0]].reshape(heads, batch_size, -1, max_len)
        t_attention = entity_attention[:, _hts[..., 1]].reshape(heads, batch_size, -1, max_len)
        context_attention = torch.sum(h_attention * t_attention, dim=0)
        relation_attention = self.attention(relation, context, input_mask)

        context_attention = context_attention.unsqueeze(dim=1)
        relation_attention = relation_attention.unsqueeze(dim=2)

        final_attention = context_attention * relation_attention

        final_attention = final_attention / (torch.sum(final_attention, dim=-1, keepdim=True) + 5.96e-8)

        context_info = final_attention @ context.unsqueeze(1)
        return context_info * ht_mask.unsqueeze(1)

    def forward(self, **param):
        input_id = param['input_id']
        input_mask = param['input_mask']
        hts = param['hts']
        mention_map = param['mention_map']
        entity_map = param['entity_map']

        relation = torch.arange(start=0, end=97, dtype=torch.int32).to(input_id)
        relation = self.relation_emb(relation).unsqueeze(0).repeat(input_id.shape[0], 1, 1)

        ht_mask = (hts.sum(-1) != 0).unsqueeze(-1)

        context, attention = process_long_input(self.PTM, input_id, input_mask, [101], [102])
        h, t = self.get_ht(context, mention_map, entity_map, hts, ht_mask)

        entity_map = entity_map @ mention_map
        context_info = self.context_pooling(context, attention, entity_map, hts, ht_mask, relation, input_mask)

        h = self.h_dense(h).unsqueeze(1)
        t = self.t_dense(t).unsqueeze(1)
        h = torch.tanh(h + self.hc_dense(context_info))
        t = torch.tanh(t + self.tc_dense(context_info))

        output = self.clas(h, t)

        output = output.squeeze(-1).permute(0, 2, 1)
        return output


class group_biLinear(nn.Module):
    def __init__(self, in_feature, out_feature, block_size):
        super(group_biLinear, self).__init__()
        assert in_feature % block_size == 0
        self.linear = nn.Linear(in_feature * block_size, out_feature)
        self.block_size = block_size

    def forward(self, input1, input2):
        B, R, N, H = input1.shape
        input1 = input1.reshape(B, R, N, H // self.block_size, self.block_size)
        input2 = input2.reshape(B, R, N, H // self.block_size, self.block_size)

        output = input1.unsqueeze(-1) * input2.unsqueeze(-2)
        output = output.reshape(B, R, N, H * self.block_size)

        return self.linear(output)
