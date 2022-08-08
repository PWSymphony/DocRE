import torch.nn as nn
import torch
import torch.nn.functional as F


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


if __name__ == "__main__":
    MY_DCGCN = DCGCN(layers=2, mem_dim=40, dropout=0.5)
    adj = torch.randn(30, 30, 30)
    gcn_inputs = torch.randn(30, 30, 40)
    print(MY_DCGCN(adj, gcn_inputs).shape)

