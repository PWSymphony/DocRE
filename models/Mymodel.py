import torch
import torch.nn as nn
from .long_BERT import process_long_input
from .group_biLinear import group_biLinear
from torch.nn.utils.rnn import pad_sequence
from functools import partial

ad_sequence = partial(pad_sequence, batch_first=True)


class my_model(nn.Module):
    def __init__(self, config, PTM):
        super(my_model, self).__init__()
        self.PTM = PTM.requires_grad_(bool(config.pre_lr))
        PTM_hidden_size = self.PTM.config.hidden_size
        block_size = 64

        self.h_dense = nn.Linear(PTM_hidden_size, PTM_hidden_size)
        self.t_dense = nn.Linear(PTM_hidden_size, PTM_hidden_size)
        self.hc_dense = nn.Linear(PTM_hidden_size, PTM_hidden_size)
        self.tc_dense = nn.Linear(PTM_hidden_size, PTM_hidden_size)
        self.clas = group_biLinear(PTM_hidden_size, config.relation_num, block_size)

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

    def forward(self, param):
        input_id = param['input_id']
        input_mask = param['input_mask']
        hts = param['hts']
        mention_map = param['mention_map']
        entity_map = param['entity_map']

        context, attention = process_long_input(self.PTM, input_id, input_mask, [101], [102])
        h, t, entity = self.get_ht(context, mention_map, entity_map, hts)
        context_info = self.context_pooling(context, attention, mention_map, entity_map, hts)

        h = torch.tanh(self.h_dense(h) + self.hc_dense(context_info))
        t = torch.tanh(self.t_dense(t) + self.tc_dense(context_info))

        return self.clas(h, t)
