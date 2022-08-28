import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from .group_biLinear import group_biLinear
from .long_BERT import process_long_input


class my_model3(nn.Module):
    def __init__(self, config, PTM):
        super(my_model3, self).__init__()
        self.PTM = PTM.requires_grad_(bool(config.pre_lr))
        PTM_hidden_size = self.PTM.config.hidden_size
        block_size = 64

        self.h_dense = nn.Linear(PTM_hidden_size, PTM_hidden_size)
        self.t_dense = nn.Linear(PTM_hidden_size, PTM_hidden_size)
        self.hc_dense = nn.Linear(PTM_hidden_size, PTM_hidden_size)
        self.tc_dense = nn.Linear(PTM_hidden_size, PTM_hidden_size)
        self.clas = group_biLinear(PTM_hidden_size, config.relation_num, block_size)

        self.g_model = GraphModel(edge_type=['inter', 'intra'], in_feature=97, out_feature=97)

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

    def forward(self, batch, *args, **kwargs):
        input_id = batch['input_id']
        input_mask = batch['input_mask']
        hts = batch['hts']
        mention_map = batch['mention_map']
        entity_map = batch['entity_map']

        graphs = batch['graphs']
        inter_index = batch['inter_index']
        intra_index = batch['intra_index']

        ht_mask = (hts.sum(-1) != 0).unsqueeze(-1)

        context, attention = process_long_input(self.PTM, input_id, input_mask, [101], [102])
        h, t, entity = self.get_ht(context, mention_map, entity_map, hts, ht_mask)

        entity_map = entity_map @ mention_map
        context_info = self.context_pooling(context, attention, entity_map, hts, ht_mask)

        h = torch.tanh(self.h_dense(h) + self.hc_dense(context_info))
        t = torch.tanh(self.t_dense(t) + self.tc_dense(context_info))
        pred = self.clas(h, t)

        inter_feature = torch.cat([pred[i, inter_index[i]] for i in range(len(pred))], dim=0)
        intra_feature = torch.cat([pred[i, intra_index[i]] for i in range(len(pred))], dim=0)
        entity_num = entity_map.sum(-1).bool().sum(-1)
        node_feature = torch.cat([entity[i, :entity_num[i]] for i in range(len(pred))])
        graphs = dgl.batch(graphs)
        graphs.edges['inter'].data['e'] = inter_feature
        graphs.edges['intra'].data['e'] = intra_feature
        graphs.nodes['entity'].data['n'] = node_feature

        graphs = self.g_model(graphs)
        graphs = dgl.unbatch(graphs)

        res = []
        for i in range(input_id.shape[0]):
            temp = torch.cat([graphs[i].edata['e'][('entity', 'inter', 'entity')],
                              graphs[i].edata['e'][('entity', 'intra', 'entity')]], dim=0)
            index = torch.tensor(inter_index[i] + intra_index[i]).sort()[1]
            temp = temp[index]
            res.append(temp)

        res = nn.utils.rnn.pad_sequence(res, batch_first=True)

        return res


class GraphModel(nn.Module):
    def __init__(self, in_feature, out_feature, edge_type):
        super(GraphModel, self).__init__()
        self.bl = nn.Bilinear(768, out_feature, 1, bias=False)
        self.bl1 = nn.Bilinear(out_feature * 2, out_feature * 2, out_feature)
        self.edge_type = edge_type
        self.funcs = {}

        for e_type in edge_type:
            self.funcs[e_type] = (fn.copy_e('e', 'm'),
                                  lambda x: {'relation': self.node_reduce(x.data['n'], x.mailbox['m'], self.bl)})
        self.e_funcs = {}
        for e_type in edge_type:
            self.e_funcs[e_type] = lambda x: {'e': self.edge_fun(x)}

    @staticmethod
    def node_reduce(src, e_info, model):
        src = src.unsqueeze(1).expand(src.shape[0], e_info.shape[1], src.shape[1])
        score = F.softmax(model(src, e_info), dim=1)
        e_info = (e_info * score).sum(1)
        return e_info

    def edge_fun(self, edge):
        src = edge.src['relation']
        src = src.reshape(src.shape[0], -1)

        dst = edge.dst['relation']
        dst = dst.reshape(dst.shape[0], -1)

        new_rel = self.bl1(src, dst)
        new_rel = edge.data['e'] + F.sigmoid(new_rel) * edge.data['e']
        return {'e': new_rel}

    def forward(self, graph: dgl.DGLHeteroGraph):
        graph.multi_update_all(self.funcs, 'stack')
        for e_type in self.edge_type:
            graph.apply_edges(self.edge_fun, etype=e_type)

        return graph
