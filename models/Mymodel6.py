from functools import partial

import dgl
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .long_BERT import process_long_input

pad_sequence = partial(pad_sequence, batch_first=True)


class group_biLinear(nn.Module):
    def __init__(self, in_feature, out_feature, block_size):
        super(group_biLinear, self).__init__()
        assert in_feature % block_size == 0
        self.linear = nn.Linear(in_feature * block_size, out_feature)
        self.block_size = block_size
        self.head = in_feature // block_size

    def forward(self, input1, input2):
        input1 = input1.reshape(-1, self.head, self.block_size)
        input2 = input2.reshape(-1, self.head, self.block_size)

        output = input1.unsqueeze(-1) * input2.unsqueeze(-2)
        output = output.reshape(output.shape[0], -1)

        return self.linear(output)


class my_model6(nn.Module):
    def __init__(self, config, PTM):
        super(my_model6, self).__init__()
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

        return torch.cat(h, dim=0), torch.cat(t, dim=0), entity

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

        return torch.cat(context_info, dim=0)

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
        self_loop_feature = torch.zeros_like(node_feature, device=node_feature.device, dtype=torch.float32)
        node_feature, edge_feature = self.EGATConv(graphs, node_feature, torch.cat([edge_feature, self_loop_feature],
                                                                                   dim=0))
        context_info = get_res(edge_feature, edge_num, other_feature, restore_index)
        new_h, new_t = get_new_ht(node_feature, hts, [len(x) for x in entity])
        new_h = torch.tanh(self.h_dense(torch.cat((new_h, context_info), dim=-1)))
        new_t = torch.tanh(self.t_dense(torch.cat((new_t, context_info), dim=-1)))
        relation_res = self.relation_clas(new_h, new_t)

        return bin_res, relation_res


def create_graph(bin_res, hts, context_info, entity):
    batch_size = len(hts)
    entity_num = [len(x) for x in entity]
    split_index = [x.shape[1] for x in hts]
    edge_index = bin_res.argmax(dim=-1).bool().detach()
    edge_index = torch.split(edge_index, split_index, dim=0)
    context_info = torch.split(context_info, split_index, dim=0)

    graphs = []
    edge_feature = []
    other_feature = []
    restore_index = []
    edge_num = []
    for i in range(batch_size):
        cur_edge_index = edge_index[i].nonzero()[:, 0]
        no_edge_index = (~edge_index[i]).nonzero()[:, 0]
        temp_index = torch.cat((cur_edge_index, no_edge_index), dim=0)
        restore_index.append(temp_index.argsort(dim=0))

        edge_list = hts[i][:, cur_edge_index]
        edge_num.append(edge_list.shape[-1])
        graph = dgl.graph((edge_list[0], edge_list[1]), num_nodes=entity_num[i])
        edge_feature.append(context_info[i][cur_edge_index])
        other_feature.append(context_info[i][no_edge_index])
        graphs.append(graph)
    return \
        graphs, torch.cat(entity, dim=0), torch.cat(edge_feature, dim=0), \
        other_feature, edge_num, edge_index, restore_index


def get_res(edge_feature, edge_num, other_feature, restore_index):
    batch_size = len(other_feature)
    edge_feature = edge_feature.reshape(edge_feature.shape[0], -1)
    edge_num.append(int(edge_feature.shape[0]) - sum(edge_num))
    edge_feature = torch.split(edge_feature, edge_num, dim=0)
    res = []
    for i in range(batch_size):
        temp = torch.cat((edge_feature[i], other_feature[i]), dim=0)
        temp = temp[restore_index[i]]

        res.append(temp)
    return torch.cat(res, dim=0)


def get_new_ht(node_feature, hts, entity_num):
    node_feature = node_feature.reshape(node_feature.shape[0], -1)
    entity = torch.split(node_feature, entity_num, dim=0)
    new_h, new_t = [], []
    for i in range(len(entity_num)):
        new_h.append(entity[i][hts[i][0], :])
        new_t.append(entity[i][hts[i][1], :])

    return torch.cat(new_h, dim=0), torch.cat(new_t, dim=0)
