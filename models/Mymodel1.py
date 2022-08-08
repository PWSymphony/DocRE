import torch
import torch.nn as nn
import torch.nn.functional as F
from .long_BERT import process_long_input
from .group_biLinear import group_biLinear
from torch.nn.utils.rnn import pad_sequence
from .GNNs import DCGCN
from .LSTM import BiLSTM


def pad(inputs, pad_value=0):
    return pad_sequence(inputs, batch_first=True, padding_value=pad_value)


class my_model1(nn.Module):
    def __init__(self, config, PTM):
        # 加上了mention pair
        super(my_model1, self).__init__()
        self.PTM = PTM.requires_grad_(bool(config.pre_lr))
        PTM_hidden_size = self.PTM.config.hidden_size
        hidden_size = config.hidden_size
        block_size = 64

        self.h_dense = nn.Linear(PTM_hidden_size * 2, PTM_hidden_size)
        self.t_dense = nn.Linear(PTM_hidden_size * 2, PTM_hidden_size)

        self.bili = group_biLinear(PTM_hidden_size, hidden_size, block_size)

        self.clas = nn.Linear(2 * hidden_size, config.relation_num)

    @staticmethod
    def get_mention_ht(context, mention_map, mention_ht, mention_ht_mask):
        batch_size = context.shape[0]
        mention = mention_map @ context

        h = torch.stack([mention[i, mention_ht[i, :, 0]] for i in range(batch_size)]) * mention_ht_mask
        t = torch.stack([mention[i, mention_ht[i, :, 1]] for i in range(batch_size)]) * mention_ht_mask

        return h, t

    @staticmethod
    def get_entity_ht(context, mention_map, entity_map, hts, ht_mask):
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

    @staticmethod
    def mention2entity(mention_logit, mention_ht_num):
        max_num = mention_logit.shape[1]
        mention_ht_num = [m + [max_num - sum(m)] for m in mention_ht_num]
        output = []
        for m, num in zip(mention_logit, mention_ht_num):
            temp = list(torch.split(m, num)[:-1])
            for index in range(len(temp)):
                temp[index] = torch.mean(temp[index], dim=0)
            output.append(torch.stack(temp, dim=0))

        return pad(output)

    def forward(self, **param):
        input_id = param['input_id']
        input_mask = param['input_mask']
        hts = param['hts']
        mention_map = param['mention_map']
        entity_map = param['entity_map']
        mention_ht = param['mention_ht']
        mention_ht_map = param['mention_ht_map']
        context, attention = process_long_input(self.PTM, input_id, input_mask, [101], [102])

        entity_ht_mask = (torch.sum(hts, dim=-1, keepdim=True) != 0)
        entity_h, entity_t = self.get_entity_ht(context, mention_map, entity_map, hts, entity_ht_mask)
        entity_context_info = self.context_pooling(context, attention, entity_map @ mention_map, hts, entity_ht_mask)

        mention_ht_mask = (torch.sum(mention_ht, dim=-1, keepdim=True) != 0)
        mention_h, mention_t = self.get_mention_ht(context, mention_map, mention_ht, mention_ht_mask)
        mention_context_info = self.context_pooling(context, attention, mention_map, mention_ht, mention_ht_mask)

        entity_h = torch.tanh(self.h_dense(torch.cat([entity_h, entity_context_info], dim=-1)))
        entity_t = torch.tanh(self.h_dense(torch.cat([entity_t, entity_context_info], dim=-1)))
        entity_ht = self.bili(entity_h, entity_t)

        mention_h = torch.tanh(self.h_dense(torch.cat([mention_h, mention_context_info], dim=-1)))
        mention_t = torch.tanh(self.h_dense(torch.cat([mention_t, mention_context_info], dim=-1)))
        mention_ht = self.bili(mention_h, mention_t)

        mention_ht = (mention_ht_map @ mention_ht) / (torch.sum(mention_ht_map, dim=-1, keepdim=True) + 1e-20)

        output = torch.cat([entity_ht, mention_ht], dim=-1)
        output = self.clas(torch.relu(output))

        return output


class sub_model(nn.Module):
    def __init__(self, in_feature):
        super(sub_model, self).__init__()
        self.dense1 = nn.Linear(4 * in_feature, in_feature)
        self.dense2 = nn.Linear(in_feature, 2)

    def forward(self, h_mention, t_mention):
        output = self.dense1(torch.cat([h_mention, t_mention, h_mention - t_mention, h_mention * t_mention], dim=-1))
        output = self.dense2(torch.relu(output))
        return output
