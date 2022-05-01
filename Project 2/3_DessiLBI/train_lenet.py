import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import * 
from collections import namedtuple
import matplotlib.pyplot as plt 
from optim.slbi_sgd import SLBI_SGD_ToolBox
from optim.slbi_adam import SLBI_ADAM_ToolBox

from loaders import *
from Models import Net
from plot import *
#使用DessiLBI训练MNIST手写数字识别

# loading
_,_,train_loader,test_loader,model_dir=loading(name='mnist',BATCH=128)

def get_slbi(model,lr,kappa=1,mu=20):
    layer_list = []
    name_list = []
    for name, p in model.named_parameters():
        name_list.append(name)
        if len(p.data.size()) == 4 or len(p.data.size()) == 2:
            layer_list.append(name)
    #定义SLBI优化器
    optimizer = SLBI_SGD_ToolBox(model.parameters(), lr=lr, kappa=kappa, mu=mu, weight_decay=0)
    optimizer.assign_name(name_list)
    optimizer.initialize_slbi(layer_list)
    return optimizer 

def train(lr,kappa,mu,interval,epoch=20,use='DessiLBI'):
    train_accs  = []
    test_accs = []
    #共训练20轮次
    for ep in range(epoch):
        #学习率衰减
        lr = lr * (0.1 ** (ep //interval))
        if use=='DessiLBI':
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        #训练
        loss_val = 0
        correct = 0
        num = 0
        model.train()
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

        train_acc = get_accuracy(train_loader)
        test_acc = get_accuracy(test_loader)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if ep  % 1 == 0:
            print('epoch', ep , 'loss', loss_val, 'acc', test_acc)
            correct = num = 0
            loss_val = 0
        #更新剪枝顺序
        if use=='DessiLBI':
            optimizer.update_prune_order(ep)
    return train_accs,test_accs


def get_accuracy(test_loader):
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

# prune the third conv layer(1) or conv3 plus fc1(2)
def prune_result(ratio,mode=1):
    if mode==1:
        optimizer.prune_layer_by_order_by_list(ratio, 'conv3.weight', True)   
    elif mode==2: 
        optimizer.prune_layer_by_order_by_list(ratio, ['conv3.weight', 'fc1.weight'], True)  
    prun_acc = get_accuracy(test_loader)
    optimizer.recover()
    return prun_acc




if __name__=='__main__':
    # 初始参数设置
    lr = 0.1
    kappa = 1
    mu = 20
    interval = 20
    epoch=30

    model = Net().to(device)
    optimizer = get_slbi(model,lr=lr,kappa=kappa,mu=mu)

    train_accs,test_accs=train(lr,kappa,mu,interval,epoch=epoch)
    torch.save(model.state_dict(), model_dir+'/train_lenet_DessiLBI.pth')
    exp_path = exp_dir+'/train_lenet_DessiLBI.png'
    plot_test_train_acc(train_accs,test_accs,'LeNet on MNIST using DessiLBI',exp_path,[0.95,1])

    # conv3权重可视化
    weight = model.conv3.weight.clone().detach().cpu().numpy()
    H=10;W=10
    weights = np.zeros((H*5,W*5))
    for i in range(H):
        for j in range(W):
            weights[i*5:i*5+5, j*5:j*5+5] = weight[i][j]
    weights = np.abs(weights)
    plot_weight_conv3(weights,title='conv3 weight by DessiLBI',savename='conv3_weight_by_DessiLBI')
    

    print('=======original accuracy:{0:.4f}======='.format(get_accuracy(test_loader)))
    ratios = [5,10,20, 40, 60, 80]
    print('prune conv3:')
    for ratio in ratios:
        prun_acc = prune_result(ratio)
        print("ratio:{0}\tpruned accuracy:{1:.4f}".format(ratio, prun_acc))
    print('prune conv3 and fc1:')
    for ratio in ratios:
        prun_acc = prune_result(ratio,mode=2)
        print("ratio:{0}\tpruned accuracy:{1:.4f}".format(ratio, prun_acc))


    # 跟SGD对比
    model = Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    train_accs,test_accs=train(lr,kappa,mu,interval,epoch=epoch,use='SGD')
    torch.save(model.state_dict(), model_dir+'/train_lenet_SGD.pth')
    exp_path = exp_dir+'/train_lenet_SGD.png'
    plot_test_train_acc(train_accs,test_accs,'LeNet on MNIST using SGD',exp_path,[0.95,1])
    # conv3权重可视化
    weight = model.conv3.weight.clone().detach().cpu().numpy()
    H=10;W=10
    weights = np.zeros((H*5,W*5))
    for i in range(H):
        for j in range(W):
            weights[i*5:i*5+5, j*5:j*5+5] = weight[i][j]
    weights = np.abs(weights)
    plot_weight_conv3(weights,title='conv3 weight by SGD',savename='conv3_weight_by_SGD')