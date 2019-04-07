import torch
import torch.nn as nn
from torch.autograd import  Variable
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo


class LSTMBranch(nn.Module):
    def __init__(self, batch_size, ip_size=300, op_size=256):
        super(LSTMBranch, self).__init__()

        self.batch_size = batch_size
        self.ip_size = ip_size
        self.op_size = op_size

        self.lstm = nn.LSTM(ip_size, op_size)

    def forward(self, ip_matrix, use_gpu=False):
        ip_matrix = ip_matrix.permute(1, 0, 2)
        ip_matrix.requires_grad= False
        h_0 = Variable(torch.zeros(1, self.batch_size, self.op_size))
        c_0 = Variable(torch.zeros(1, self.batch_size, self.op_size))

        # Move to GPU if CUDA is available
        if torch.cuda.is_available() and use_gpu==True:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        op, _ = self.lstm(ip_matrix, (h_0, c_0))
        op = op.permute(1, 0, 2)
        return op


class VGG19(nn.Module):
    def __init__(self, embedding_dim=256, pretrained=True):
        super(VGG19, self).__init__()
        seed_model = imagemodels.__dict__['vgg19'](pretrained=pretrained).features
        seed_model = nn.Sequential(*list(seed_model.children())[:-1]) # remove final maxpool
        last_layer_index = len(list(seed_model.children()))
        seed_model.add_module(str(last_layer_index),nn.Conv2d(512, embedding_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
        self.image_model = seed_model

    def forward(self, x):
        x = self.image_model(x)
        return x

