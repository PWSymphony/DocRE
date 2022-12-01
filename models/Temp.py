import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .utils import MIN, group_biLinear, process_long_input


class Temp(nn.Module):
    def __init__(self, args):
        super(Temp, self).__init__()
        self.bert = AutoModel.from_pretrained(args.bert_name).requires_grad_(bool(args.pre_lr))
        tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
        self.cls_token_id = [tokenizer.cls_token_id]
        self.sep_token_id = [tokenizer.sep_token_id]

        bert_hidden_size = self.bert.config.hidden_size
        block_size = 64

        self.h_dense = nn.Linear(bert_hidden_size * 2, bert_hidden_size)
        self.t_dense = nn.Linear(bert_hidden_size * 2, bert_hidden_size)
        self.clas = group_biLinear(bert_hidden_size, args.relation_num, block_size)
        self.cls = Cls(args, 2)

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

        return h, t

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

        context, attention = process_long_input(self.bert, input_id, input_mask, self.cls_token_id, self.sep_token_id)
        h, t = self.get_ht(context, mention_map, entity_map, hts)
        context_info = self.context_pooling(context, attention, mention_map, entity_map, hts)

        # bin_h = torch.tanh(self.bin_h_dense(torch.cat([h, context_info], dim=-1)))
        # bin_t = torch.tanh(self.bin_t_dense(torch.cat([h, context_info], dim=-1)))
        # bin_res = self.bin_clas(bin_h, bin_t)
        # # bin_emb = F.softmax(bin_res, dim=-1) @ self.bin_emb

        h = torch.tanh(self.h_dense(torch.cat([h, context_info], dim=-1)))
        t = torch.tanh(self.t_dense(torch.cat([h, context_info], dim=-1)))
        res = self.clas(h, t)

        return res, self.cls(**kwargs)


class Cls(nn.Module):
    def __init__(self, args, num_class=2):
        super(Cls, self).__init__()
        # self.bert = AutoModel.from_pretrained(args.bert_name).requires_grad_(bool(args.pre_lr))
        self.bert = AutoModel.from_pretrained(args.bert_name).requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id

        bert_hidden_size = self.bert.config.hidden_size
        block_size = 64

        self.h_dense = nn.Linear(bert_hidden_size * 2, bert_hidden_size)
        self.t_dense = nn.Linear(bert_hidden_size * 2, bert_hidden_size)
        self.clas = group_biLinear(bert_hidden_size, num_class, block_size)

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

        return h, t

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

        context, attention = process_long_input(self.bert, input_id, input_mask, self.cls_token_id, self.sep_token_id)
        h, t = self.get_ht(context, mention_map, entity_map, hts)
        context_info = self.context_pooling(context, attention, mention_map, entity_map, hts)

        h = torch.tanh(self.h_dense(torch.cat([h, context_info], dim=-1)))
        t = torch.tanh(self.t_dense(torch.cat([h, context_info], dim=-1)))
        res = self.clas(h, t)

        return res