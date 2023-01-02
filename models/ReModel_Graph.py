from functools import partial

import dgl
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer

from . import MIN, group_biLinear, process_long_input

pad_sequence = partial(pad_sequence, batch_first=True)


class ReModel_Graph(nn.Module):
    def __init__(self, args):
        super(ReModel_Graph, self).__init__()
        self.bert = AutoModel.from_pretrained(args.bert_name).requires_grad_(bool(args.pre_lr))
        tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
        self.cls_token_id = [tokenizer.cls_token_id]
        self.sep_token_id = [tokenizer.sep_token_id]

        bert_hidden_size = self.bert.config.hidden_size
        head = 12
        # self.type_emb = nn.Embedding(37, 768, padding_idx=36)
        # self.edge_info = FFNN(bert_hidden_size * 2, bert_hidden_size)
        self.gat = GATs(in_feature=bert_hidden_size, out_feature=bert_hidden_size // head, heads=head, k=2)

        self.info = FFNN(bert_hidden_size * 2, bert_hidden_size)
        self.classify = Classify(bert_hidden_size, args.relation_num)

    # @staticmethod
    # def get_ht(context, mention_map, entity_map, hts):
    #     batch_size = context.shape[0]
    #     entity_mask = torch.sum(entity_map, dim=-1, keepdim=True) == 0
    #     mention = mention_map @ context
    #     mention_mean = mention.mean(dim=-1, keepdim=True)
    #     mention = mention - mention_mean
    #     mention = torch.exp(mention)
    #     mention_mean = torch.exp(mention_mean)
    #     entity = entity_map @ mention
    #     entity_mean = entity_map @ mention_mean
    #     entity = torch.log(entity) + torch.log(entity_mean)
    #     h = torch.stack([entity[i, hts[i, :, 0]] for i in range(batch_size)])
    #     t = torch.stack([entity[i, hts[i, :, 1]] for i in range(batch_size)])
    #
    #     return h, t, entity

    @staticmethod
    def get_ht(context, mention_map, entity_map, hts):
        batch_size = context.shape[0]
        entity_mask = torch.sum(entity_map, dim=-1, keepdim=True) == 0
        mention = mention_map @ context
        mention = torch.exp(mention)
        entity = entity_map @ mention
        entity = torch.masked_fill(entity, entity_mask, 1)
        entity = torch.log(entity)
        h = torch.stack([entity[i, hts[i, :, 0]] for i in range(batch_size)])
        t = torch.stack([entity[i, hts[i, :, 1]] for i in range(batch_size)])

        return h, t, entity

    @staticmethod
    def context_pooling(context, attention, mention_map, entity_map, hts):
        batch_size = context.shape[0]
        e_map = entity_map @ mention_map
        e_map = e_map / (torch.sum(e_map, dim=-1, keepdim=True) + MIN)
        entity_attention = (e_map.unsqueeze(1) @ attention)
        h_attention = torch.stack([entity_attention[i][:, hts[i, :, 0]] for i in range(batch_size)], dim=0)
        t_attention = torch.stack([entity_attention[i][:, hts[i, :, 1]] for i in range(batch_size)], dim=0)
        context_attention = torch.sum(h_attention * t_attention, dim=1)
        context_attention = context_attention / (torch.sum(context_attention, dim=-1, keepdim=True) + MIN)
        context_info = context_attention @ context

        return context_info

    def forward(self, **kwargs):
        input_id = kwargs['input_id']
        input_mask = kwargs['input_mask']
        hts = kwargs['hts']
        mention_map = kwargs['mention_map']
        entity_map = kwargs['entity_map']
        # type_pair = kwargs['type_pair']
        ht_num = kwargs['relation_mask'].sum(-1)
        entity_num = entity_map.sum(-1).bool().sum(-1).long()

        context, attention = process_long_input(self.bert, input_id, input_mask, self.cls_token_id, self.sep_token_id)
        h, t, entity = self.get_ht(context, mention_map, entity_map, hts)
        context_info = self.context_pooling(context, attention, mention_map, entity_map, hts)

        # type_info = self.type_emb(type_pair)
        # ht_info = self.edge_info(torch.cat([context_info, type_info], dim=-1))
        ht_info = torch.cat([context_info[i, :x] for i, x in enumerate(ht_num)], dim=0)
        entity = torch.cat([entity[i, :x] for i, x in enumerate(entity_num)], dim=0)
        graph = dgl.batch(kwargs['graphs'])

        node_info, edge_info = self.gat(graph, entity, ht_info)
        edge_info = torch.split(edge_info, list(ht_num), dim=0)
        edge_info = pad_sequence(edge_info)

        info = self.info(torch.cat([edge_info, context_info], dim=-1))

        result = self.classify(h, t, info)

        return {'pred': result}


class GAT(nn.Module):
    def __init__(self, in_feature, out_feature, heads):
        super().__init__()
        self.GAT = dgl.nn.EGATConv(in_node_feats=in_feature, in_edge_feats=in_feature,
                                   out_node_feats=out_feature, out_edge_feats=out_feature, num_heads=heads)
        self.edge_dense = nn.Linear(heads * out_feature, out_feature * heads)
        self.node_dense = nn.Linear(heads * out_feature, out_feature * heads)
        self.edge_layer_norm = nn.LayerNorm(out_feature * heads)
        self.node_layer_norm = nn.LayerNorm(out_feature * heads)

    def forward(self, graph, node_feature, edge_feature):
        node, edge = self.GAT(graph, node_feature, edge_feature)
        edge = torch.relu(self.edge_dense(edge.reshape(edge.shape[0], -1)))
        node = torch.relu(self.node_dense(node.reshape(node.shape[0], -1)))
        edge = self.edge_layer_norm(edge)
        node = self.edge_layer_norm(node)
        return node, edge


class GATs(nn.Module):
    def __init__(self, in_feature, out_feature, heads, k=1):
        super(GATs, self).__init__()
        self.k = k
        self.nets = nn.ModuleList([GAT(in_feature, out_feature, heads) for _ in range(k)])

    def forward(self, graph, node_feature, edge_feature):
        node, edge = node_feature, edge_feature
        for net in self.nets:
            node, edge = net(graph, node, edge)
            node = node + node_feature
            edge = edge + edge_feature

        return node, edge


class FFNN(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(FFNN, self).__init__()
        self.dense1 = nn.Linear(in_feature, out_feature * 3)
        self.dense2 = nn.Linear(out_feature * 3, out_feature)

    def forward(self, src):
        return self.dense2(torch.relu(self.dense1(src)))


class Classify(nn.Module):
    def __init__(self, hidden_size, num_class=2, block_size=64):
        super(Classify, self).__init__()
        self.h_dense = nn.Linear(hidden_size, hidden_size)
        self.t_dense = nn.Linear(hidden_size, hidden_size)
        self.hc_dense = nn.Linear(hidden_size, hidden_size)
        self.tc_dense = nn.Linear(hidden_size, hidden_size)
        self.clas = group_biLinear(hidden_size, num_class, block_size)

    def forward(self, h, t, context_info):
        h = torch.tanh(self.h_dense(h) + self.hc_dense(context_info))
        t = torch.tanh(self.t_dense(t) + self.hc_dense(context_info))
        res = self.clas(h, t)

        return res
