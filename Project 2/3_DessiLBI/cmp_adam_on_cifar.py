import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import * 
from collections import namedtuple
import matplotlib.pyplot as plt 
from optim.slbi_adam import SLBI_ADAM_ToolBox
import numpy as np 

from loaders import Seek_dir,loading
from Models import Net,ResNet
from plot import *
#使用DessiLBI训练CIFAR10图像分类任务
torch.manual_seed(42)

def get_accuracy(model,test_loader):
    model.eval()
    correct = 0
    num = 0
    for iter, pack in enumerate(test_loader):
        data, target = pack[0].to(device), pack[1].to(device)
        logits = model(data)
        _, pred = logits.max(1)
        correct += pred.eq(target).sum().item()
        num += data.shape[0]
    acc = correct / num 
    return acc 

def get_slbi(model,lr,kappa=1,mu=20):
    layer_list = []
    name_list = []
    for name, p in model.named_parameters():
        name_list.append(name)
        if len(p.data.size()) == 4 or len(p.data.size()) == 2:
            layer_list.append(name)
    #定义SLBI优化器
    optimizer = SLBI_ADAM_ToolBox(model.parameters(), lr=lr, kappa=kappa, mu=mu, weight_decay=0)
    optimizer.assign_name(name_list)
    optimizer.initialize_slbi(layer_list)
    return optimizer 

def train_one_epoch(model,train_loader):
    model.train()
    num = 0
    loss_val = 0
    for iter, pack in enumerate(train_loader):
        data, target = pack[0].to(device), pack[1].to(device)
        logits = model(data)
        loss = F.nll_loss(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, pred = logits.max(1)
        loss_val += loss.item()
        num += data.shape[0]
    loss_val /= num 
    train_acc=get_accuracy(model,train_loader)
    return loss_val ,train_acc

if __name__=='__main__':
    # loading
    _,_,train_loader,test_loader,model_dir=loading(name='cifar',BATCH=128)


    losses_adam = []
    losses_adam_slbi = []
    accs_adam = []
    accs_adam_slbi = []
    traccs_adam_slbi=[]
    traccs_adam=[]
    LR = 3e-4
    EPOCH = 20

    #使用adam版本的lsbi训练
    model = ResNet().to(device)
    optimizer = get_slbi(model,lr=LR)

    for ep in range(EPOCH):
        loss,tracc = train_one_epoch(model,train_loader)
        test_acc = get_accuracy(model,test_loader)
        losses_adam_slbi.append(loss)
        accs_adam_slbi.append(test_acc)
        traccs_adam_slbi.append(tracc)
        if ep  % 1 == 0:
            print('epoch', ep , 'loss', loss, 'accuracy', test_acc)
        
        #optimizer.update_prune_order(ep)

    adam_weight3_1 = model.conv3_1.weight.clone().detach().cpu().numpy()
    adam_weight3_2 = model.conv3_2.weight.clone().detach().cpu().numpy()

    #使用torch的adam训练
    model = ResNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(EPOCH):
        loss,tracc = train_one_epoch(model,train_loader)
        test_acc = get_accuracy(model,test_loader)
        losses_adam.append(loss)
        accs_adam.append(test_acc)
        traccs_adam.append(tracc)
        if ep  % 1 == 0:
            print('epoch', ep , 'loss', loss, 'accuracy', test_acc)

    slbi_weight3_1 = model.conv3_1.weight.clone().detach().cpu().numpy()
    slbi_weight3_2 = model.conv3_2.weight.clone().detach().cpu().numpy()


    # 模型loss与acc对比
    cmp_loss_and_acc(loss1=losses_adam_slbi,loss2=losses_adam,
                    acc1=accs_adam_slbi,acc2=accs_adam,
                    model1='Adam in slbi',model2='Original Adam',
                    savename='cmp_loss_and_acc')


    # 组内精度对比
    compare_accs(traccs_adam_slbi,accs_adam_slbi,traccs_adam,accs_adam,
                'Adam in slbi','Original Adam',exp_dir+'/compare_acc_of_adam.png',ylim=[0,1])


    #对比两者的权重
    H = 10
    W = 10
    adam_weight = np.zeros((H*3,W*3))
    for i in range(H):
        for j in range(W):
            adam_weight[i*3:i*3+3, j*3:j*3+3] = adam_weight3_1[i][j]
    adam_weight = np.abs(adam_weight)

    slbi_weight = np.zeros((H*3,W*3))
    for i in range(H):
        for j in range(W):
            slbi_weight[i*3:i*3+3, j*3:j*3+3] = slbi_weight3_1[i][j]
    slbi_weight = np.abs(slbi_weight)

    plot_compare_weight_conv3(adam_weight,'adam',slbi_weight,'slbi','cmp_adam_slbi_weight')