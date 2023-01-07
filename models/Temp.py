import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .utils import GroupBiLinear, context_pooling, get_ht, process_long_input


class Temp(nn.Module):
    def __init__(self, args):
        super(Temp, self).__init__()
        self.bert = AutoModel.from_pretrained(args.bert_name).requires_grad_(bool(args.pre_lr))
        tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
        self.cls_token_id = [tokenizer.cls_token_id]
        self.sep_token_id = [tokenizer.sep_token_id]
        block_size = 64

        self.classify = Classify(self.bert.config.hidden_size, args.relation_num, block_size)
        # self.classify_bin = Classify(self.bert.config.hidden_size, 2, block_size)

    def forward(self, **kwargs):
        input_id = kwargs['input_id']
        input_mask = kwargs['input_mask']
        hts = kwargs['hts']
        mention_map = kwargs['mention_map']
        entity_map = kwargs['entity_map']

        context, attention = process_long_input(self.bert, input_id, input_mask, self.cls_token_id, self.sep_token_id)
        h, t, _ = get_ht(context, mention_map, entity_map, hts)
        context_info = context_pooling(context, attention, mention_map, entity_map, hts)

        res = self.classify(h, t, context_info)
        res_bin = self.classify_bin(h, t, context_info)

        return {'pred': res, 'res_bin': res_bin}


class Classify(nn.Module):
    def __init__(self, hidden_size, num_class=2, block_size=64):
        super(Classify, self).__init__()
        self.h_dense = nn.Linear(hidden_size, hidden_size)
        self.t_dense = nn.Linear(hidden_size, hidden_size)
        self.hc_dense = nn.Linear(hidden_size, hidden_size)
        self.tc_dense = nn.Linear(hidden_size, hidden_size)
        self.clas = GroupBiLinear(hidden_size, num_class, block_size)

    def forward(self, h, t, context_info):
        h = torch.tanh(self.h_dense(h) + self.hc_dense(context_info))
        t = torch.tanh(self.t_dense(t) + self.hc_dense(context_info))
        res = self.clas(h, t)

        return res
