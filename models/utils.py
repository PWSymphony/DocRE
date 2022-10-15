import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MIN = 2 ** (-14)
MAX = 65500


class group_biLinear(nn.Module):
    def __init__(self, in_feature, out_feature, block_size):
        super(group_biLinear, self).__init__()
        assert in_feature % block_size == 0
        self.linear = nn.Linear(in_feature * block_size, out_feature)
        self.block_size = block_size

    def forward(self, input1, input2):
        B, L, H = input1.shape
        input1 = input1.reshape(B, L, H // self.block_size, self.block_size)
        input2 = input2.reshape(B, L, H // self.block_size, self.block_size)

        output = input1.unsqueeze(-1) * input2.unsqueeze(-2)
        output = output.reshape(B, L, H * self.block_size)

        return self.linear(output)


def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens):
    # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
    # 例如输入长度最大为5，当输入长度为7：[1,2,3,4,5,6,7]时，分成[1,2,3,4,5]和[3,4,5,6,7]两部分

    n, c = input_ids.size()
    start_tokens = torch.tensor(start_tokens).to(input_ids)
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)
    if c <= 512:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
    else:
        new_input_ids, new_attention_mask, num_seg = [], [], []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
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
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
        i = 0
        new_output, new_attention = [], []
        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:
                output = F.pad(sequence_output[i], (0, 0, 0, c - 512))
                att = F.pad(attention[i], (0, c - 512, 0, c - 512))
                new_output.append(output)
                new_attention.append(att)
            elif n_s == 2:
                output1 = sequence_output[i][:512 - len_end]
                mask1 = attention_mask[i][:512 - len_end]
                att1 = attention[i][:, :512 - len_end, :512 - len_end]
                output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                mask1 = F.pad(mask1, (0, c - 512 + len_end))
                att1 = F.pad(att1, (0, c - 512 + len_end, 0, c - 512 + len_end))

                output2 = sequence_output[i + 1][len_start:]
                mask2 = attention_mask[i + 1][len_start:]
                att2 = attention[i + 1][:, len_start:, len_start:]
                output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                att2 = F.pad(att2, [l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i])
                mask = mask1 + mask2 + 1e-10
                output = (output1 + output2) / mask.unsqueeze(-1)
                att = (att1 + att2)
                att = att / (att.sum(-1, keepdim=True) + 1e-10)  # 归一化
                new_output.append(output)
                new_attention.append(att)
            i += n_s
        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)
    return sequence_output, attention


class BiLSTM(nn.Module):
    def __init__(self, input_size, config, hidden_size=None, dropout=None):
        super().__init__()
        self.config = config
        if not hidden_size:
            hidden_size = config.hidden_size
        if not dropout:
            dropout = config.dropout
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.in_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths, pad_value=0):
        """
        src: [batch_size, slen, input_size]
        src_lengths: [batch_size]
        """

        src = self.in_dropout(src)

        packed_src = nn.utils.rnn.pack_padded_sequence(src, src_lengths.cpu(), batch_first=True,
                                                       enforce_sorted=False)
        packed_outputs, _ = self.lstm(packed_src)
        packed_outputs, _ = self.lstm(packed_src)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, padding_value=pad_value)

        return outputs


class DCGCN(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, mem_dim, layers, dropout, self_loop=True):
        super(DCGCN, self).__init__()
        assert mem_dim % layers == 0
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(p=dropout)

        # linear transformation
        self.linear_output = nn.Linear(self.mem_dim, self.mem_dim)

        # DCGCN block
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))

        # self.weight_list = self.weight_list.cuda()
        # self.linear_output = self.linear_output.cuda()
        self.self_loop = self_loop

    def forward(self, adj, gcn_inputs):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            if self.self_loop:
                AxW = AxW + self.weight_list[l](outputs)  # self loop
            else:
                AxW = AxW

            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs

        out = self.linear_output(gcn_outputs)

        return out
