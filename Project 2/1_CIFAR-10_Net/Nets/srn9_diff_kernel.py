from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU,Mish,ELU,RReLU,GELU

class cbr(torch.nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1, bn=True, activation='ReLU'):
        super(cbr, self).__init__()
        op = [
                nn.Conv2d(channels_in, channels_out,
                                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        ]
        if bn:
            op.append(nn.BatchNorm2d(channels_out))
        if activation in ['ReLU','Mish','ELU','RReLU',]:
            op.append(globals()[activation](inplace=True))
        if activation in ['GELU']:
            op.append(globals()[activation]())
        self.layer = nn.Sequential(*op)

    def forward(self, x):
        return self.layer(x)

class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class AMP2d(torch.nn.Module):
    def __init__(self,H=1,W=1):
        super(AMP2d, self).__init__()
        self.layer = nn.AdaptiveMaxPool2d((H,W))

    def forward(self, x):
        return self.layer(x)

class simp_ResNet9_k7(nn.Module):
    def __init__(self, num_classes=10, if_bn=True, channel=64,activation='ReLU'):
        super(simp_ResNet9_k7, self).__init__()
        self.activation=activation
        self.layer=nn.Sequential(
            cbr(3, 128, kernel_size=7, stride=2, padding=3,activation=activation),
            # torch.nn.MaxPool2d(2),

            Residual(nn.Sequential(
                cbr(128, 32,kernel_size=1,padding=0,activation=activation),
                cbr(32, 32,activation=activation),
                cbr(32, 128,kernel_size=1,padding=0,activation=activation),
            )),

            cbr(128, 256, kernel_size=3, stride=1, padding=1,activation=activation),
            nn.MaxPool2d(2),

            Residual(nn.Sequential(
                cbr(256, 64,kernel_size=1,padding=0,activation=activation),
                cbr(64, 64,activation=activation),
                cbr(64, 256,kernel_size=1,padding=0,activation=activation),
            )),

            cbr(256, 128, kernel_size=3, stride=1, padding=0,activation=activation),

            AMP2d(1,1),
            Flatten(),
            nn.Linear(128, num_classes, bias=False),
        )
    def forward(self,x):
        out=self.layer(x)
        return out

class simp_ResNet9_k333(nn.Module):
    def __init__(self, num_classes=10, if_bn=True, channel=64,activation='ReLU'):
        super(simp_ResNet9_k333, self).__init__()
        self.activation=activation
        self.layer=nn.Sequential(
            cbr(3, 32, kernel_size=3, stride=1, padding=1,activation=activation),
            cbr(32, 64, kernel_size=3, stride=1, padding=2,activation=activation),
            cbr(64, 128, kernel_size=3, stride=2, padding=2,activation=activation),
            # torch.nn.MaxPool2d(2),

            Residual(nn.Sequential(
                cbr(128, 32,kernel_size=1,padding=0,activation=activation),
                cbr(32, 32,activation=activation),
                cbr(32, 128,kernel_size=1,padding=0,activation=activation),
            )),

            cbr(128, 256, kernel_size=3, stride=1, padding=1,activation=activation),
            nn.MaxPool2d(2),

            Residual(nn.Sequential(
                cbr(256, 64,kernel_size=1,padding=0,activation=activation),
                cbr(64, 64,activation=activation),
                cbr(64, 256,kernel_size=1,padding=0,activation=activation),
            )),

            cbr(256, 128, kernel_size=3, stride=1, padding=0,activation=activation),

            AMP2d(1,1),
            Flatten(),
            nn.Linear(128, num_classes, bias=False),
        )
    def forward(self,x):
        out=self.layer(x)
        return out

class simp_ResNet9_k33(nn.Module):
    def __init__(self, num_classes=10, if_bn=True, channel=64,activation='ReLU'):
        super(simp_ResNet9_k33, self).__init__()
        self.activation=activation
        self.layer=nn.Sequential(
            cbr(3, 64, kernel_size=3, stride=1, padding=1,activation=activation),
            cbr(64, 128, kernel_size=3, stride=2, padding=2,activation=activation),
            # torch.nn.MaxPool2d(2),

            Residual(nn.Sequential(
                cbr(128, 32,kernel_size=1,padding=0,activation=activation),
                cbr(32, 32,activation=activation),
                cbr(32, 128,kernel_size=1,padding=0,activation=activation),
            )),

            cbr(128, 256, kernel_size=3, stride=1, padding=1,activation=activation),
            nn.MaxPool2d(2),

            Residual(nn.Sequential(
                cbr(256, 64,kernel_size=1,padding=0,activation=activation),
                cbr(64, 64,activation=activation),
                cbr(64, 256,kernel_size=1,padding=0,activation=activation),
            )),

            cbr(256, 128, kernel_size=3, stride=1, padding=0,activation=activation),

            AMP2d(1,1),
            Flatten(),
            nn.Linear(128, num_classes, bias=False),
        )
    def forward(self,x):
        out=self.layer(x)
        return out