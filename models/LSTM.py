import torch.nn as nn


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
        # self.lstm.flatten_parameters()  # 为了多卡运行rnn
        # bsz, slen, input_size = src.size()

        src = self.in_dropout(src)

        packed_src = nn.utils.rnn.pack_padded_sequence(src, src_lengths.cpu(), batch_first=True,
                                                       enforce_sorted=False)
        # packed_outputs, (src_h_t, src_c_t) = self.lstm(packed_src)
        packed_outputs, _ = self.lstm(packed_src)
        packed_outputs, _ = self.lstm(packed_src)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, padding_value=pad_value)

        # 先将多层lstm得到的h和c的（D * n_layer, ..）分开为（n_layer, D, ..）
        # src_h_t = src_h_t.reshape(self.config.nlayers, 2, bsz, self.hidden_size)
        # src_c_t = src_c_t.reshape(self.config.nlayers, 2, bsz, self.hidden_size)
        # # 取最后一层的h和c，并将正向和反向的输出进行拼接
        # output_h_t = torch.cat((src_h_t[-1, 0], src_h_t[-1, 1]), dim=-1)
        # output_c_t = torch.cat((src_c_t[-1, 0], src_c_t[-1, 1]), dim=-1)

        # outputs = self.out_dropout(outputs) # 在Mymodel13中去掉
        # output_h_t = self.out_dropout(output_h_t)
        # output_c_t = self.out_dropout(output_c_t)

        return outputs
