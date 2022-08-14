import torch
import torch.nn as nn
import dgl
from .long_BERT import process_long_input
from .group_biLinear import group_biLinear


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

        entity_map = entity_map / (torch.sum(entity_map, dim=-1, keepdim=True) + 1e-20)
        entity_attention = (entity_map.unsqueeze(1) @ attention).permute(1, 0, 2, 3)

        entity_attention = entity_attention.reshape(heads, -1, max_len)
        h_attention = entity_attention[:, _hts[..., 0]].reshape(heads, batch_size, -1, max_len)
        t_attention = entity_attention[:, _hts[..., 1]].reshape(heads, batch_size, -1, max_len)
        context_attention = torch.sum(h_attention * t_attention, dim=0)
        context_attention = context_attention / (torch.sum(context_attention, dim=-1, keepdim=True) + 1e-20)

        context_info = context_attention @ context
        return context_info * ht_mask

    def forward(self, batch):
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
        h, t = self.get_ht(context, mention_map, entity_map, hts, ht_mask)

        entity_map = entity_map @ mention_map
        context_info = self.context_pooling(context, attention, entity_map, hts, ht_mask)

        h = torch.tanh(self.h_dense(h) + self.hc_dense(context_info))
        t = torch.tanh(self.t_dense(t) + self.tc_dense(context_info))
        pred = self.clas(h, t)

        inter_feature = torch.cat([pred[i, inter_index[i]] for i in range(len(pred))], dim=0)
        intra_feature = torch.cat([pred[i, intra_index[i]] for i in range(len(pred))], dim=0)
        graphs = dgl.batch(graphs)
        graphs.edges['inter'].data['e'] = inter_feature
        graphs.edges['intra'].data['e'] = intra_feature

        return pred
