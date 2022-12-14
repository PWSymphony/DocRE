from functools import partial

import dgl
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from .group_biLinear import group_biLinear
from .long_BERT import process_long_input


pad_sequence = partial(pad_sequence, batch_first=True)


class my_model5(nn.Module):
    def __init__(self, config, PTM):
        super(my_model5, self).__init__()
        self.PTM = PTM.requires_grad_(bool(config.pre_lr))
        PTM_hidden_size = self.PTM.config.hidden_size
        block_size = 64
        assert PTM_hidden_size % block_size == 0

        self.hb_dense = nn.Linear(PTM_hidden_size * 2, PTM_hidden_size)
        self.tb_dense = nn.Linear(PTM_hidden_size * 2, PTM_hidden_size)

        self.bin_clas = group_biLinear(PTM_hidden_size, 2, block_size)
        self.relation_clas = group_biLinear(PTM_hidden_size, config.relation_num, block_size)
        self.h_dense = nn.Linear(PTM_hidden_size * 2, PTM_hidden_size)
        self.t_dense = nn.Linear(PTM_hidden_size * 2, PTM_hidden_size)

        self.EGATConv = dgl.nn.pytorch.EGATConv(in_node_feats=PTM_hidden_size,
                                                in_edge_feats=PTM_hidden_size,
                                                out_node_feats=block_size,
                                                out_edge_feats=block_size,
                                                num_heads=PTM_hidden_size // block_size)

    @staticmethod
    def get_ht(context, mention_map, entity_map, hts):
        batch_size = context.shape[0]

        entity = []
        h = []
        t = []
        for i in range(batch_size):
            mention = context[i][mention_map[i]]
            mention = torch.exp(mention)
            cur_entity = torch.log(entity_map[i] @ mention)
            entity.append(cur_entity)
            h.append(cur_entity[hts[i][0], :])
            t.append(cur_entity[hts[i][1], :])

        return pad_sequence(h), pad_sequence(t), entity

    @staticmethod
    def context_pooling(context, attention, mention_map, entity_map, hts):
        batch_size = attention.shape[0]

        context_info = []
        for i in range(batch_size):
            cur_attention = entity_map[i] @ attention[i][:, mention_map[i]]
            cur_attention = cur_attention / (torch.sum(entity_map[i], dim=-1, keepdim=True) + 1e-20)
            h_attention = cur_attention[:, hts[i][0]]
            t_attention = cur_attention[:, hts[i][1]]
            context_attention = torch.sum(h_attention * t_attention, dim=0)
            context_attention = torch.div(context_attention,
                                          (torch.sum(context_attention, dim=-1, keepdim=True) + 1e-20))
            context_info.append(context_attention @ context[i])

        return pad_sequence(context_info)

    def forward(self, param, **kwargs):
        input_id = param['input_id']
        input_mask = param['input_mask']
        hts = param['hts']
        mention_map = param['mention_map']
        entity_map = param['entity_map']

        context, attention = process_long_input(self.PTM, input_id, input_mask, [101], [102])
        h, t, entity = self.get_ht(context, mention_map, entity_map, hts)
        context_info = self.context_pooling(context, attention, mention_map, entity_map, hts)

        bin_h = torch.tanh(self.hb_dense(torch.cat((h, context_info), dim=-1)))
        bin_t = torch.tanh(self.tb_dense(torch.cat((t, context_info), dim=-1)))
        bin_res = self.bin_clas(bin_h, bin_t)

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

        return bin_res, relation_res


# def create_graph(bin_res, hts, context_info, entity):
#     batch_size = context_info.shape[0]
#     entity_num = [len(x) for x in entity]
#     edge_index = (bin_res[..., 1] > bin_res[..., 0]).cpu().numpy()  # >=
#
#     graphs = []
#     edge_feature = []
#     edge_num = []
#     for i in range(batch_size):
#         edge_index[i][len(hts[0][0]):] = False
#         edge_list = hts[i][:, edge_index[i, :hts[i].shape[-1]].tolist()]
#         edge_num.append(edge_list.shape[-1])
#         graph = dgl.graph((edge_list[0], edge_list[1]), num_nodes=entity_num[i])
#         edge_feature.append(context_info[i][edge_index[i]])
#         graphs.append(graph.to(bin_res.device))
#     return graphs, torch.cat(entity, dim=0), torch.cat(edge_feature, dim=0), edge_num, edge_index

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


# def get_res(edge_feature, context_info, edge_num, edge_index):
#     edge_feature = edge_feature.reshape(edge_feature.shape[0], -1)
#     edge_num.append(int(edge_feature.shape[0]) - sum(edge_num))
#     edge_feature = torch.split(edge_feature, edge_num, dim=0)
#     res = []
#     for i in range(context_info.shape[0]):
#         temp = []
#         e_index = 0
#         for j, k in enumerate(edge_index[i]):
#             if k:
#                 temp.append(edge_feature[i][e_index])
#                 e_index += 1
#             else:
#                 temp.append(context_info[i][j])
#         temp = torch.stack(temp, dim=0)
#         res.append(temp)
#     return pad_sequence(res)

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