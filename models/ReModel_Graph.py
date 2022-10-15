from functools import partial

import dgl
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .utils import MIN, group_biLinear, process_long_input

pad_sequence = partial(pad_sequence, batch_first=True)


class ReModel_Graph(nn.Module):
    def __init__(self, config, bert, cls_token_id, sep_token_id):
        super(ReModel_Graph, self).__init__()
        self.bert = bert.requires_grad_(bool(config.pre_lr))
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        bert_hidden_size = self.bert.config.hidden_size
        block_size = 64

        self.hb_dense = nn.Linear(bert_hidden_size * 2, bert_hidden_size)
        self.tb_dense = nn.Linear(bert_hidden_size * 2, bert_hidden_size)

        self.bin_clas = group_biLinear(bert_hidden_size, 2, block_size)
        self.relation_clas = group_biLinear(bert_hidden_size, config.relation_num, block_size)
        self.h_dense = nn.Linear(bert_hidden_size * 2, bert_hidden_size)
        self.t_dense = nn.Linear(bert_hidden_size * 2, bert_hidden_size)

        self.EGATConv = dgl.nn.pytorch.EGATConv(in_node_feats=bert_hidden_size,
                                                in_edge_feats=bert_hidden_size,
                                                out_node_feats=block_size,
                                                out_edge_feats=block_size,
                                                num_heads=bert_hidden_size // block_size)

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

        return h, t, entity

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
        h, t, entity = self.get_ht(context, mention_map, entity_map, hts, ht_mask)

        entity_map = entity_map @ mention_map
        context_info = self.context_pooling(context, attention, entity_map, hts, ht_mask)

        graphs, node_feature, edge_feature, other_feature, edge_num, edge_index, restore_index \
            = create_graph(bin_res, hts, context_info, entity)

        graphs = dgl.batch(graphs)
        graphs = dgl.add_self_loop(graphs)
        node_feature, edge_feature = self.EGATConv(graphs, node_feature, torch.cat([edge_feature, node_feature], dim=0))
        context_info = get_res(edge_feature, context_info, edge_num, other_feature, restore_index)
        new_h, new_t = get_new_ht(node_feature, hts, [len(x) for x in entity])
        new_h = torch.tanh(self.h_dense(torch.cat((new_h, context_info), dim=-1)))
        new_t = torch.tanh(self.t_dense(torch.cat((new_t, context_info), dim=-1)))
        relation_res = self.relation_clas(new_h, new_t)

        return relation_res


def create_graph(bin_res, hts, context_info, entity):
    batch_size = context_info.shape[0]
    entity_num = [len(x) for x in entity]
    edge_index = bin_res.argmax(dim=-1).bool().cpu().numpy()

    graphs = []
    edge_feature = []
    other_feature = []
    restore_index = []
    edge_num = []
    for i in range(batch_size):
        edge_index[i][len(hts[0][0]):] = False
        index = np.concatenate((edge_index[i].nonzero()[0], (~edge_index[i]).nonzero()[0]), axis=0)
        restore_index.append(np.argsort(index, axis=-1))
        edge_list = hts[i][:, edge_index[i, :hts[i].shape[-1]].tolist()]
        edge_num.append(edge_list.shape[-1])
        graph = dgl.graph((edge_list[0], edge_list[1]), num_nodes=entity_num[i])
        edge_feature.append(context_info[i][edge_index[i]])
        other_feature.append(context_info[i][~edge_index[i]])
        graphs.append(graph.to(bin_res.device))
    return \
        graphs, torch.cat(entity, dim=0), torch.cat(edge_feature, dim=0), \
        other_feature, edge_num, edge_index, restore_index


def get_res(edge_feature, context_info, edge_num, other_feature, restore_index):
    batch_size = len(other_feature)
    edge_feature = edge_feature.reshape(edge_feature.shape[0], -1)
    edge_num.append(int(edge_feature.shape[0]) - sum(edge_num))
    edge_feature = torch.split(edge_feature, edge_num, dim=0)
    res = []
    for i in range(batch_size):
        temp = torch.cat((edge_feature[i], other_feature[i]), dim=0)
        temp = temp[restore_index[i]]

        res.append(temp)
    return torch.stack(res, dim=0)


def get_new_ht(node_feature, hts, entity_num):
    node_feature = node_feature.reshape(node_feature.shape[0], -1)
    entity = torch.split(node_feature, entity_num, dim=0)
    new_h, new_t = [], []
    for i in range(len(entity_num)):
        new_h.append(entity[i][hts[i][0], :])
        new_t.append(entity[i][hts[i][1], :])

    return pad_sequence(new_h), pad_sequence(new_t)
