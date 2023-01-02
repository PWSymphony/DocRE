import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .utils import MAX, MIN, group_biLinear, process_long_input


class ReModel_Mention(nn.Module):
    def __init__(self, args):
        super(ReModel_Mention, self).__init__()
        self.bert = AutoModel.from_pretrained(args.bert_name).requires_grad_(bool(args.pre_lr))
        tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
        self.cls_token_id = [tokenizer.cls_token_id]
        self.sep_token_id = [tokenizer.sep_token_id]
        bert_hidden_size = self.bert.config.hidden_size
        block_size = 64

        self.bert_emb = self.bert.get_input_embeddings()
        self.h_dense = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.t_dense = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.hc_dense = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.tc_dense = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.clas = group_biLinear(bert_hidden_size, args.relation_num, block_size)

    # @staticmethod
    # def get_ht(context, attention, mention_map, entity_map, hts, ht_mask):
    #     batch_size = context.shape[0]
    #     ht_num = hts.sum(-1).bool().sum(-1)
    #     mention_num = entity_map.sum(-1)
    #     mention_att = mention_map @ attention.mean(dim=1) @ mention_map.permute(0, 2, 1)
    #     mention = mention_map @ context
    #     entity = entity_map @ mention
    #
    #     h, t = [], []
    #     for b in range(batch_size):
    #         cur_h, cur_t = [], []
    #         for ht in range(ht_num[b]):
    #             if mention_num[b, hts[b, ht, 0]] == 1 and mention_num[b, hts[b, ht, 1]] == 1:
    #                 cur_h.append(entity[b, hts[b, ht, 0]])
    #                 cur_t.append(entity[b, hts[b, ht, 1]])
    #             else:
    #                 h_index = entity_map[b, hts[b, ht, 0]].nonzero().squeeze(-1)
    #                 t_index = entity_map[b, hts[b, ht, 1]].nonzero().squeeze(-1)
    #                 cur_att = mention_att[b][h_index][:, t_index]
    #                 h_t = cur_att.sum(1) / (cur_att.sum() + MIN)
    #                 t_h = cur_att.sum(0) / (cur_att.sum() + MIN)
    #                 h_mention = h_t @ mention[b][h_index]
    #                 t_mention = t_h @ mention[b][t_index]
    #                 cur_h.append(h_mention)
    #                 cur_t.append(t_mention)
    #         h.append(F.pad(torch.stack(cur_h, dim=0), pad=(0, 0, 0, max(ht_num) - ht_num[b])))
    #         t.append(F.pad(torch.stack(cur_t, dim=0), pad=(0, 0, 0, max(ht_num) - ht_num[b])))
    #
    #     return torch.stack(h, dim=0), torch.stack(t, dim=0)
    @staticmethod
    def get_ht(context, attention, mention_map, entity_map, hts, ht_mask):
        batch_size = context.shape[0]
        mention = mention_map @ context
        mention_att = mention_map @ attention.sum(dim=1) @ mention_map.permute(0, 2, 1)
        entity_att = entity_map @ mention_att
        h_mask = torch.stack([entity_map[i, hts[i, :, 0]] for i in range(batch_size)]) * ht_mask
        t_mask = torch.stack([entity_map[i, hts[i, :, 1]] for i in range(batch_size)]) * ht_mask
        h_att = torch.stack([entity_att[i, hts[i, :, 0]] for i in range(batch_size)]) * t_mask
        t_att = torch.stack([entity_att[i, hts[i, :, 1]] for i in range(batch_size)]) * h_mask

        h_att = h_att / (torch.sum(h_att, dim=-1, keepdim=True) + MIN)
        t_att = t_att / (torch.sum(t_att, dim=-1, keepdim=True) + MIN)
        h = (t_att @ mention) * ht_mask
        t = (h_att @ mention) * ht_mask

        return h, t, h_mask, t_mask

    @staticmethod
    def context_pooling(context, attention, entity_map, hts, ht_mask):
        batch_size, max_len, _ = context.shape
        heads = attention.shape[1]

        max_entity = entity_map.shape[1]
        _hts = hts.clone()
        for i in range(batch_size):
            _hts[i] = _hts[i] + i * max_entity
        _hts = _hts.reshape(-1, 2)

        entity_map = entity_map / (torch.sum(entity_map, dim=-1, keepdim=True) + MIN)
        entity_attention = (entity_map.unsqueeze(1) @ attention).permute(1, 0, 2, 3)

        entity_attention = entity_attention.reshape(heads, -1, max_len)
        h_attention = entity_attention[:, _hts[..., 0]].reshape(heads, batch_size, -1, max_len)
        t_attention = entity_attention[:, _hts[..., 1]].reshape(heads, batch_size, -1, max_len)
        context_attention = torch.sum(h_attention * t_attention, dim=0)
        context_attention = context_attention / (torch.sum(context_attention, dim=-1, keepdim=True) + MIN)

        context_info = context_attention @ context
        return context_info * ht_mask

    def forward(self, **kwargs):
        input_id = kwargs['input_id']
        input_mask = kwargs['input_mask']
        hts = kwargs['hts']
        mention_map = kwargs['mention_map']
        entity_map = kwargs['entity_map']

        ht_mask = (hts.sum(-1) != 0).unsqueeze(-1)

        context, attention = process_long_input(self.bert, input_id, input_mask, self.cls_token_id, self.sep_token_id)
        h, t, h_mask, t_mask = self.get_ht(context, attention, mention_map, entity_map, hts, ht_mask)

        entity_map = entity_map @ mention_map
        context_info = self.context_pooling(context, attention, entity_map, hts, ht_mask)

        h_info = self.hc_dense(context_info)
        t_info = self.tc_dense(context_info)

        h = torch.tanh(self.h_dense(h) + h_info)
        t = torch.tanh(self.t_dense(t) + t_info)
        res = self.clas(h, t)

        type_mask = torch.zeros((input_mask.shape[0], 4),
                                device=input_mask.device,
                                dtype=input_id.dtype)
        type_mask = torch.cat((type_mask, input_mask), dim=-1)
        attention_mask = torch.ones((input_mask.shape[0], 4),
                                    device=input_mask.device,
                                    dtype=input_id.dtype)
        attention_mask = torch.cat((attention_mask, input_mask), dim=-1)
        sep_emb = torch.tensor([self.sep_token_id], dtype=input_id.dtype,
                               device=input_id.device).repeat(input_id.shape[0], 1)
        input_emb = self.bert_emb(torch.cat((input_id[:, :1], sep_emb, input_id[:, 1:]), dim=-1)).unsqueeze(1)
        input_emb = input_emb.repeat(1, h.shape[1], 1, 1)
        input_emb = torch.cat([input_emb[:, :, :1], h.unsqueeze(2), t.unsqueeze(2), context_info.unsqueeze(2),
                               input_emb[:, :, 1:]], dim=-2)
        # res = res - (~kwargs['type_mask']).float() * MAX
        return {'pred': res}


def process_long_input_emb(model, input_ids, attention_mask, type_mask, start_tokens, end_tokens):
    n, c, h = input_ids.size()
    start_tokens = torch.tensor(start_tokens).to(input_ids)
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)
    if c <= 512:
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = output[0]
    else:
        new_input_ids, new_attention_mask, num_seg = [], [], []
        seq_len = attention_mask.sum(1).cpu().tolist()
        for i, l_i in enumerate(seq_len):
            if l_i <= 512:
                new_input_ids.append(input_ids[i, :512])
                new_attention_mask.append(attention_mask[i, :512])
                num_seg.append(1)
            else:
                input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
                input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
                attention_mask1 = attention_mask[i, :512]
                attention_mask2 = attention_mask[i, (l_i - 512): l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attention_mask.extend([attention_mask1, attention_mask2])
                num_seg.append(2)
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = output[0]
        i = 0
        new_output = []
        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:
                output = F.pad(sequence_output[i], (0, 0, 0, c - 512))
                new_output.append(output)
            elif n_s == 2:
                output1 = sequence_output[i][:512 - len_end]
                mask1 = attention_mask[i][:512 - len_end]
                output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                mask1 = F.pad(mask1, (0, c - 512 + len_end))

                output2 = sequence_output[i + 1][len_start:]
                mask2 = attention_mask[i + 1][len_start:]
                output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                mask = mask1 + mask2 + 1e-10
                output = (output1 + output2) / mask.unsqueeze(-1)
                new_output.append(output)
            i += n_s
        sequence_output = torch.stack(new_output, dim=0)
    return sequence_output


class MentionAttention(nn.Module):
    def __init__(self, args, hidden_size):
        super(MentionAttention, self).__init__()
        self.q_dense = nn.Linear(args.relation_num, hidden_size)
        self.k_dense = nn.Linear(hidden_size, hidden_size)
        self.head = 1
        self.scale = (hidden_size / self.head) ** -0.5

    def forward(self, res, mention, h_mask, t_mask):
        # q = self.q_dense(res).reshape(res.shape[0], res.shape[1], self.head, -1)
        # k = self.k_dense(mention).reshape(mention.shape[0], mention.shape[1], self.head, -1)
        # q = q.permute(0, 2, 1, 3)
        # k = k.permute(0, 2, 3, 1)

        q = self.q_dense(res)
        k = self.k_dense(mention).permute(0, 2, 1)
        att = q @ k * self.scale
        h_att = att - (1 - t_mask) * MAX
        t_att = att - (1 - h_mask) * MAX

        h_att = F.softmax(h_att, dim=-1)
        t_att = F.softmax(t_att, dim=-1)

        h = (t_att @ mention)
        t = (h_att @ mention)

        return h, t
