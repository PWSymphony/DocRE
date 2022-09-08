import dgl
import torch
import torch.nn as nn

from .group_biLinear import group_biLinear
from .long_BERT import process_long_input


class my_model5(nn.Module):

    def __init__(self, config, PTM):
        super(my_model5, self).__init__()
        self.PTM = PTM.requires_grad_(bool(config.pre_lr))
        PTM_hidden_size = self.PTM.config.hidden_size
        block_size = 64
        assert PTM_hidden_size % block_size == 0
        self.h_dense = nn.Linear(PTM_hidden_size, PTM_hidden_size)
        self.t_dense = nn.Linear(PTM_hidden_size, PTM_hidden_size)
        self.hc_dense = nn.Linear(PTM_hidden_size, PTM_hidden_size)
        self.tc_dense = nn.Linear(PTM_hidden_size, PTM_hidden_size)
        self.bin_clas = group_biLinear(PTM_hidden_size, 2, block_size)
        self.EGATConv = dgl.nn.pytorch.EGATConv(in_node_feats=PTM_hidden_size,
                                                in_edge_feats=PTM_hidden_size,
                                                out_node_feats=block_size,
                                                out_edge_feats=block_size,
                                                num_heads=PTM_hidden_size // block_size)

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
        entity_map = entity_map / (torch.sum(entity_map, dim=-1, keepdim=True) + 1e-20)
        entity_attention = (entity_map.unsqueeze(1) @ attention).permute(1, 0, 2, 3)
        entity_attention = entity_attention.reshape(heads, -1, max_len)
        h_attention = entity_attention[:, _hts[..., 0]].reshape(heads, batch_size, -1, max_len)
        t_attention = entity_attention[:, _hts[..., 1]].reshape(heads, batch_size, -1, max_len)
        context_attention = torch.sum(h_attention * t_attention, dim=0)
        context_attention = context_attention / (torch.sum(context_attention, dim=-1, keepdim=True) + 1e-20)
        context_info = context_attention @ context
        return context_info * ht_mask

    def forward(self, param, **kwargs):
        input_id = param['input_id']
        input_mask = param['input_mask']
        hts = param['hts']
        mention_map = param['mention_map']
        entity_map = param['entity_map']
        ht_mask = (hts.sum(-1) != 0).unsqueeze(-1)
        context, attention = process_long_input(self.PTM, input_id, input_mask, [101], [102])
        h, t, entity = self.get_ht(context, mention_map, entity_map, hts, ht_mask)
        entity_map = entity_map @ mention_map
        context_info = self.context_pooling(context, attention, entity_map, hts, ht_mask)
        h = torch.tanh(self.h_dense(h) + self.hc_dense(context_info))
        t = torch.tanh(self.t_dense(t) + self.tc_dense(context_info))
        bin_res = self.bin_clas(h, t)
        graphs, node_feature, edge_feature, edge_num, edge_index = create_graph(bin_res, hts, context_info, entity)
        graphs = dgl.batch(graphs)
        graphs = dgl.add_self_loop(graphs)
        node_feature, edge_feature = self.EGATConv(graphs, node_feature, torch.cat([edge_feature, node_feature], dim=0))
        return bin_res


def create_graph(bin_res, hts, context_info, entity):
    batch_size = hts.shape[0]
    entity_num = (1 + hts.max(-1)[0].max(-1)[0]).tolist()
    edge_index = bin_res[..., 1] > bin_res[..., 0]
    edge_list = hts[edge_index]
    edge_num = edge_index.sum(-1).tolist()
    edge = edge_list.split(edge_num, dim=0)
    graphs = []
    edge_feature = []
    node_feature = []
    for i in range(batch_size):
        cur_edge = edge[i][edge[i].sum(-1) != 0].t()
        edge_num[i] = cur_edge.shape[1]
        graph = dgl.graph((cur_edge[0], cur_edge[1]), num_nodes=entity_num[i])
        node_feature.append(entity[i, :entity_num[i]])
        edge_feature.append(context_info[i][edge_index[i]][:cur_edge.shape[1]])
        graphs.append(graph)
    return graphs, torch.cat(node_feature, dim=0), torch.cat(edge_feature, dim=0), edge_num, edge_index

# def get_res(edge_feature, edge_index, edge_num):

