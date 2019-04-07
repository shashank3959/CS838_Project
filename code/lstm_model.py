import torch
import torch.nn as nn
from torch.autograd import  Variable


class LSTMBranch(nn.Module):
    def __init__(self, batch_size, ip_size, op_size):
        super(LSTMBranch, self).__init__()

        self.batch_size = batch_size
        self.ip_size = ip_size
        self.op_size = op_size

        self.lstm = nn.LSTM(ip_size, op_size)

    def forward(self, ip_matrix):
        ip_matrix = ip_matrix.permute(1, 0, 2)
        ip_matrix.requires_grad= False
        h_0 = Variable(torch.zeros(1, self.batch_size, self.op_size))
        c_0 = Variable(torch.zeros(1, self.batch_size, self.op_size))

        op, _ = self.lstm(ip_matrix, (h_0, c_0))
        op = op.permute(1, 0, 2)
        return op