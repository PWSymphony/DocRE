from functools import partial

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from . import MIN, group_biLinear, process_long_input

pad_sequence = partial(pad_sequence, batch_first=True)


class ReModel_Graph(nn.Module):
    def __init__(self, config, bert, cls_token_id, sep_token_id):
        super(ReModel_Graph, self).__init__()
        self.bert = bert.requires_grad_(bool(config.pre_lr))
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id

        bert_hidden_size = self.bert.config.hidden_size
        block_size = 64
        graph_feature_size = 256

        self.h_dense = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.t_dense = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.hc_dense = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.tc_dense = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.extraction = group_biLinear(bert_hidden_size, graph_feature_size, block_size)
        self.CLS_dense = nn.Linear(bert_hidden_size, graph_feature_size)

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

    def get_graph_feature(self, context, src, ht_num):
        feature = []
        for i in range(context.shape[0]):
            feature.append(self.CLS_dense(context[i, :1]))
            feature.append(src[i, :ht_num[i]])

        return torch.cat(feature, dim=0)

    @staticmethod
    def process_res(src, ht_num):
        max_ht_num = max(ht_num)
        src = torch.split(src, (ht_num + 1).tolist(), dim=0)
        res = []
        for i in range(ht_num.shape[0]):
            res.append(F.pad(src[i][1:], (0, 0, 0, max_ht_num - src[i].shape[0] + 1)))

        return torch.stack(res, dim=0)

    def forward(self, **kwargs):
        input_id = kwargs['input_id']
        input_mask = kwargs['input_mask']
        hts = kwargs['hts']
        mention_map = kwargs['mention_map']
        entity_map = kwargs['entity_map']

        ht_mask = (hts.sum(-1) != 0).unsqueeze(-1)

        context, attention = process_long_input(self.bert, input_id, input_mask, self.cls_token_id, self.sep_token_id)
        h, t = self.get_ht(context, mention_map, entity_map, hts, ht_mask)
        context_info = self.context_pooling(context, attention, entity_map @ mention_map, hts, ht_mask)

        h = torch.tanh(self.h_dense(h) + self.hc_dense(context_info))
        t = torch.tanh(self.t_dense(t) + self.tc_dense(context_info))
        res = self.extraction(h, t)
        CLS = self.CLS_dense(context[:, :1])

        return res


class GAT(nn.Module):
    def __init__(self, in_feature, out_feature, heads):
        super().__init__()
        self.GAT = dgl.nn.GATv2Conv(in_feature, out_feature, num_heads=heads, feat_drop=0.1, residual=True)
        self.dense = nn.Linear(heads * out_feature, out_feature)

    def forward(self, graph, node_feature):
        output = self.GAT(graph, node_feature)
        output = self.dense(output.reshape(output.shape[0], -1))
        return torch.relu(output)


class GATs(nn.Module):
    def __init__(self, in_feature, out_feature, heads, k=1):
        super(GATs, self).__init__()
        self.nets = nn.ModuleList([GAT(in_feature, out_feature, heads) for _ in range(k)])

    def forward(self, graph, node_feature):
        for net in self.nets:
            node_feature = net(graph, node_feature)

        return node_feature


class FFNN(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(FFNN, self).__init__()
        self.dense1 = nn.Linear(in_feature, out_feature * 3)
        self.dense2 = nn.Linear(out_feature * 3, out_feature)

    def forward(self, src):
        return self.dense2(torch.relu(self.dense1(src)))
