import torch.nn as nn


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
