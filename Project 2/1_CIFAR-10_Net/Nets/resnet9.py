from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU,Mish,ELU,RReLU,GELU
from torch.utils.tensorboard import SummaryWriter

class cbr(torch.nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1,
                 bn=True, activation='ReLU',inplace=True):
        super(cbr, self).__init__()
        op = [
                nn.Conv2d(channels_in, channels_out,
                                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        ]
        if bn:
            op.append(nn.BatchNorm2d(channels_out))
        if activation in ['ReLU','Mish','ELU','RReLU',]:
            op.append(globals()[activation](inplace=inplace))
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

class MaxDropout(nn.Module):
    def __init__(self, drop=0.3):
#         print(p)
        super(MaxDropout, self).__init__()
        if drop < 0 or drop > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(drop))
        self.drop = 1 - drop

    def forward(self, x):
        if not self.training:
            return x

        up = x - x.min()
        divisor =  (x.max() - x.min())
        x_copy = torch.div(up,divisor)
        if x.is_cuda:
            x_copy = x_copy.cuda()

        mask = (x_copy > (self.drop))
        x = x.masked_fill(mask > 0.5, 0)
        return x 
        

class ResNet9(nn.Module):
    def __init__(self, num_classes=10, if_bn=True, channel=64,inplace=True):
        super(ResNet9, self).__init__()
        self.layer=nn.Sequential(
            cbr(3, 64, kernel_size=3, stride=1, padding=1,inplace=inplace),
            cbr(64, 128, kernel_size=5, stride=2, padding=2,inplace=inplace),
            # torch.nn.MaxPool2d(2),
        
            Residual(nn.Sequential(
                cbr(128, 128,inplace=inplace),
                cbr(128, 128,inplace=inplace),
            )),

            cbr(128, 256, kernel_size=3, stride=1, padding=1,inplace=inplace),
            nn.MaxPool2d(2),

            Residual(nn.Sequential(
                cbr(256, 256,inplace=inplace),
                cbr(256, 256,inplace=inplace),
            )),

            cbr(256, 128, kernel_size=3, stride=1, padding=0,inplace=inplace),

            AMP2d(1,1),
            Flatten(),
            nn.Linear(128, num_classes, bias=False),
        )
    def forward(self,x):
        out=self.layer(x)
        return out

class ResNet9_MaxDropout(nn.Module):
    def __init__(self, num_classes=10, if_bn=True, channel=64):
        super(ResNet9_MaxDropout, self).__init__()
        self.dropout = MaxDropout()  # dropout训练
        self.layer=nn.Sequential(
            cbr(3, 64, kernel_size=3, stride=1, padding=1),
            cbr(64, 128, kernel_size=5, stride=2, padding=2),
            # torch.nn.MaxPool2d(2),

            Residual(nn.Sequential(
                cbr(128, 128),
                cbr(128, 128),
            )),
            self.dropout,
            cbr(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),

            Residual(nn.Sequential(
                cbr(256, 256),
                cbr(256, 256),
            )),
            self.dropout,
            cbr(256, 128, kernel_size=3, stride=1, padding=0),
            
            AMP2d(1,1),
            Flatten(),
            nn.Linear(128, num_classes, bias=False),
        )
    def forward(self,x):
        out=self.layer(x)
        return out
    

class simp_ResNet9(nn.Module):
    def __init__(self, num_classes=10, if_bn=True, channel=64,activation='ReLU'):
        super(simp_ResNet9, self).__init__()
        self.activation=activation
        self.layer=nn.Sequential(
            cbr(3, 64, kernel_size=3, stride=1, padding=1,activation=activation),
            cbr(64, 128, kernel_size=5, stride=2, padding=2,activation=activation),
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


class simp_ResNet9_MaxDropout(nn.Module):
    def __init__(self, num_classes=10, if_bn=True, channel=64,activation='ReLU'):
        super(simp_ResNet9_MaxDropout, self).__init__()
        self.dropout = MaxDropout()  # dropout训练
        self.layer=nn.Sequential(
            cbr(3, 64, kernel_size=3, stride=1, padding=1,activation=activation),
            cbr(64, 128, kernel_size=5, stride=2, padding=2,activation=activation),
            # torch.nn.MaxPool2d(2),

            Residual(nn.Sequential(
                cbr(128, 32,kernel_size=1,padding=0,activation=activation),
                cbr(32, 32,activation=activation),
                cbr(32, 128,kernel_size=1,padding=0,activation=activation),
            )),
            self.dropout,
            cbr(128, 256, kernel_size=3, stride=1, padding=1,activation=activation),
            nn.MaxPool2d(2),

            Residual(nn.Sequential(
                cbr(256, 64,kernel_size=1,padding=0,activation=activation),
                cbr(64, 64,activation=activation),
                cbr(64, 256,kernel_size=1,padding=0,activation=activation),
            )),
            self.dropout,
            cbr(256, 128, kernel_size=3, stride=1, padding=0,activation=activation),

            AMP2d(1,1),
            Flatten(),
            nn.Linear(128, num_classes, bias=False),
        )
    def forward(self,x):
        out=self.layer(x)
        return out


if __name__ == '__main__':
    net = ResNet9()
    input = torch.ones((64,3,32,32))
    output = net(input)
    # tensorboard
    writer = SummaryWriter("logs")
    writer.add_graph(net,input)
    writer.close()