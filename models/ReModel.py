import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .utils import GroupBiLinear, context_pooling, get_ht, process_long_input


class ReModel(nn.Module):
    def __init__(self, args):
        super(ReModel, self).__init__()
        self.bert = AutoModel.from_pretrained(args.bert_name).requires_grad_(bool(args.pre_lr))
        tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
        self.cls_token_id = [tokenizer.cls_token_id]
        self.sep_token_id = [tokenizer.sep_token_id]
        bert_hidden_size = self.bert.config.hidden_size
        block_size = 64

        self.h_dense = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.t_dense = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.hc_dense = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.tc_dense = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.clas = GroupBiLinear(bert_hidden_size, args.relation_num, block_size)

    def forward(self, **kwargs):
        input_id = kwargs['input_id']
        input_mask = kwargs['input_mask']
        hts = kwargs['hts']
        mention_map = kwargs['mention_map']
        entity_map = kwargs['entity_map']

        context, attention = process_long_input(self.bert, input_id, input_mask, self.cls_token_id, self.sep_token_id)
        h, t, entity = get_ht(context, mention_map, entity_map, hts)
        context_info = context_pooling(context, attention, mention_map, entity_map, hts)
        h = torch.tanh(self.h_dense(h) + self.hc_dense(context_info))
        t = torch.tanh(self.t_dense(t) + self.hc_dense(context_info))
        res = self.clas(h, t)

        return {'pred': res}
