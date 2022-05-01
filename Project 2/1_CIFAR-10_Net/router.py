import sched
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss,SmoothL1Loss,BCEWithLogitsLoss

from torch.utils.tensorboard import SummaryWriter

import torch.optim as optim
import time
from time import sleep
from tqdm import tqdm
import os
from pathlib import Path

from Nets import *
from Configs import args
import fire

# use GPU or CPU
device = torch.device('cuda:0')
## cudNN 基准
torch.backends.cudnn.benchmark = True

def Seek_dir(Dirname:str='JXHdatasets',plan=2,fd=Path(__file__).resolve().parent):
    '''plan--往上查找层数      fd--运行文件父路径'''
    origin_fd=fd
    while plan: 
        for p in fd.glob(Dirname): return p.as_posix()
        plan-=1 ;fd=fd.parent
    # 没找到文件夹，在爷路径下创建文件夹
    print('DirPath Not Found,creating now! Path=',origin_fd.parent.as_posix()+'/'+Dirname)
    Path(origin_fd.parent.as_posix()+'/'+Dirname).mkdir()
    return origin_fd.parent.as_posix()+'/'+Dirname  

## 按批加载好训练集和测试集
def dataloader(batchsize=128,download=True,data_path='default',Erase=False):
    '''
    batchsize:  批大小
    done: 是否已把数据下载到本地
    '''
    if Erase==True: 
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  #先四周填充0，图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            #R,G,B每层的归一化用到的均值和方差
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.4)
                            , ratio=(0.3, 3.3)
                            , value=(0.4914, 0.4822, 0.4465)), #填充
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  #先四周填充0，图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    time_start = time.time()
    works_num=4
    dataset_path=Seek_dir('JXHdatasets',3) if data_path=='default' else data_path
    print('数据集位置: ',dataset_path)
    trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True
                ,download=download, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize
                ,shuffle=True, num_workers=works_num,pin_memory=True)
    testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False
                ,download=download, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize
                ,shuffle=False, num_workers=works_num,pin_memory=True)
    print('Load Time Cost:', time.time() - time_start,)
    torch.cuda.empty_cache()
    return trainloader,testloader

## 模型权重初始化
def initial_para(net=None):
    assert net!=None,'权重初始化没有传入网络'
    for module in net.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data.fill_(1.0)
            module.eps = 0.00001
            module.momentum = 0.1
        else:
            module.float()
        if isinstance(module, torch.nn.Conv2d) and hasattr(module, 'weight'):
            # torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))  # original
            torch.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='linear')
            # torch.nn.init.xavier_uniform_(module.weight, gain=torch.nn.init.calculate_gain('linear'))
        if isinstance(module, torch.nn.Linear) and hasattr(module, 'weight'):
            # torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))  # original
            torch.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='linear')
            # torch.nn.init.xavier_uniform_(module.weight, gain=1.)
    torch.cuda.empty_cache()
    return net

## 模型准确率
def eval(net,trainset=None,testset=None,oncuda=True,SetType='test'):
    net.eval()
    trainacc=0;testacc=0
    with torch.no_grad():
        total=0;correct=0
        if trainset!=None:
            for data in trainset:
                images, labels = data
                if oncuda:
                    images = images.to(device)
                    labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            trainacc= 100 * correct / total  
        total=0;correct=0   
        if testset!=None:
            for data in testset:
                images, labels = data
                if oncuda:
                    images = images.to(device)
                    labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            testacc= 100 * correct / total  
    # print(outputs)
    torch.cuda.empty_cache()
    return trainacc,testacc

## 保存模型
def save_model(net=None,ModelName=None):
    assert ModelName!=None and net!=None, '保存模型：缺少参数'
    # fdir='../cifar10_model_data'
    fdir=Seek_dir('cifar10_model_data')
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    torch.save(net.state_dict(), fdir+'/'+ModelName)

## 读取模型
def load_model(net=None,ModelName=None):
    assert ModelName!=None and net!=None, '读取模型：缺少参数'
    # fdir='../cifar10_model_data'
    fdir=Seek_dir('cifar10_model_data')
    net.load_state_dict(torch.load(fdir+'/'+ModelName))
    return net

## 主函数封装
def main(**kwargs):
    # 调整参数
    args._parse(kwargs)
    """Load in Data"""
    trainloader,testloader=dataloader(data_path=args.datapath
                                        ,batchsize=args.batchsize
                                        ,download=args.download
                                        ,Erase=args.Erase)
    if hasattr(globals()[args.model_chosen](),'activation') and args.activation is not None:
        model=globals()[args.model_chosen](activation=args.activation)
    else:
        model=globals()[args.model_chosen]()
    model_savename=args.model_chosen+'.pth'

    if args.mode=='train':
        
        """Network Parameters"""
        print('==> Building model '+str(args.model_chosen)+'..')
        net = model.to(device)
        net=initial_para(net)
        print(net)  if args.show_net else 0
        criterion = globals()[args.loss]()
        #注意,无论optim中的lr设置是啥,最后起作用的还是max_lr
        optimizer = optim.SGD(net.parameters()
                                    , lr=args.lr, momentum=0.9
                                    ,weight_decay=args.weight_decay)
        if args.scheduler=='OneCycleLR':
            scheduler0 =optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.4,three_phase=True 
                                    ,steps_per_epoch=len(trainloader)
                                    ,epochs=args.max_epoch)
        if args.scheduler=='CyclicLR':
            scheduler0 = optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr
                                    ,step_size_up=2
                                    ,step_size_down=16
                                    , max_lr=0.2)
        if args.scheduler=='Cos':
            optimizer = optim.SGD(net.parameters(), lr=0.1
                                    , momentum=0.9
                                    ,weight_decay=args.weight_decay)
            scheduler0 =optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.max_epoch)


        """Train the Network"""        
        print('==> training '+'..')
        # loop over the dataset multiple times
        time_start = time.time()
        stage2=False
        big1=0;big2=0
        loop=tqdm(range(args.max_epoch),unit='epoch',
                bar_format='{n_fmt}/{total_fmt}|{bar}| [{rate_fmt}{postfix}]\t')  
        for epoch in loop:
            net.train()
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                if args.loss in ['BCEWithLogitsLoss','SmoothL1Loss']:
                    labels=F.one_hot(labels, num_classes=args.num_classes).float()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # learning rate schedule
                if args.scheduler in ['OneCycleLR']:
                    scheduler0.step()
            # learning rate schedule
            if args.scheduler in ['CyclicLR','Cos']:
                scheduler0.step()
            
            # evaluate current model
            trainACC,testACC=eval(net,trainloader,testloader)
            big1=max(big1,trainACC)
            big2=max(big2,testACC)

            # print info
            if args.scheduler is not None:
                loop.set_postfix_str(
                    'loss:%.2f, lr:%.5f, trainacc:%.2f, testacc:%.2f'
                    %(running_loss / len(trainloader),
                        scheduler0.get_last_lr()[0],trainACC,testACC))
            else:
                loop.set_postfix_str(
                    'loss:%.2f, lr:%.4f, trainacc=%.2f, testacc=%.2f'
                    %(running_loss / len(trainloader),
                        optimizer.state_dict()['param_groups'][0]['lr'],trainACC,testACC))
            

            # visualize
            if args.visualize and epoch%args.plot_every_n_loop==0:
                if args.log_name is None:
                    writer = SummaryWriter('./tensorboardData/'+
                                            args.model_chosen+'+'+
                                            args.activation+'+'+
                                            args.scheduler+'+'+
                                            args.loss+"/logs")
                else:
                    writer = SummaryWriter('./tensorboardData/'+args.log_name+"/logs")
                writer.add_scalars("within Acc", {'train':trainACC,
                        'test':testACC},epoch+1)
                writer.add_scalar('between Acc',testACC,epoch+1)
                writer.add_scalar('loss',loss,epoch+1)
                if args.scheduler is not None:
                    writer.add_scalar('learning rate',scheduler0.get_last_lr()[0],epoch+1)

            # adjust cyclicLR
            if args.scheduler=='CyclicLR' and stage2==False \
            and scheduler0.get_last_lr()[0]<=args.lr:
                stage2=True
                scheduler0=torch.optim.lr_scheduler.ExponentialLR(optimizer,1-1e-2)

        print('==> finished training!')
        print('Totally Training Time Cost',time.time() - time_start)
        print('Best TrainAcc','%.3f%%'%(big1),', best TestAcc','%.3f%%'%(big2), '\n')

        """Save the model"""
        save_model(net,model_savename)

        if args.visualize:
            writer.close()
        
    elif args.mode=='test':

        """Load the model"""
        net = model.to(device)
        load_model(net,model_savename)

        """Evaluate Performance"""
        trainACC,testACC=eval(net,trainloader,testloader)
        print('Accuracy of the network on train set: %.3f %%' % (trainACC))
        print('Accuracy of the network on test set: %.3f %%' % (testACC))

    ## 清除显存
    torch.cuda.empty_cache()



if __name__ == '__main__':
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
    fire.Fire()


'''(example)
python router.py main --max_epoch=1
'''


'''Stage 1: try different models'''

'''(ResNet18)
python router.py main --model_chosen=ResNet18
'''

'''(DLA)
python router.py main --model_chosen=DLA
'''

'''(ResNet9)
python router.py main --model_chosen=ResNet9
'''


'''(SimpResNet9)
python router.py main --model_chosen=simp_ResNet9
'''



'''Stage 2: learning rate schedulers'''

'''(CyclicLR)
python router.py main --model_chosen=simp_ResNet9 --scheduler='CyclicLR'
'''

'''('OneCycleLR')
python router.py main --model_chosen=simp_ResNet9 --scheduler='OneCycleLR'
'''


'''('Cos')
python router.py main --model_chosen=simp_ResNet9 --scheduler='Cos'
'''


'''Stage 3: activations'''

'''(Mish)
python router.py main --model_chosen=simp_ResNet9 --scheduler='CyclicLR' --activation='Mish'
'''


'''(ELU)
python router.py main --model_chosen=simp_ResNet9 --scheduler='CyclicLR' --activation='ELU'
'''


'''(RReLU)
python router.py main --model_chosen=simp_ResNet9 --scheduler='CyclicLR' --activation='RReLU'
'''


'''(GELU)
python router.py main --model_chosen=simp_ResNet9 --scheduler='CyclicLR' --activation='GELU'
'''



'''Stage 4: kernel size'''

'''(SimpResNet9--Firstblock[3,5])
python router.py main --model_chosen=simp_ResNet9
'''


'''(SimpResNet9--Firstblock[7])
python router.py main --model_chosen=simp_ResNet9_k7
'''


'''(SimpResNet9--Firstblock[3,3,3])
python router.py main --model_chosen=simp_ResNet9_k333
'''


'''(SimpResNet9--Firstblock[3,3])
python router.py main --model_chosen=simp_ResNet9_k33
'''


'''(SimpResNet9--Firstblock[3,3]+CyclicLR)
python router.py main --model_chosen=simp_ResNet9_k33 --scheduler='CyclicLR'
'''



'''Stage 5: loss function'''
'''default--CrossEntropyLoss'''

'''(SmoothL1Loss)
python router.py main --model_chosen=simp_ResNet9 --loss=SmoothL1Loss
'''


'''(BCEWithLogitsLoss)
python router.py main --model_chosen=simp_ResNet9 --loss=BCEWithLogitsLoss
'''



'''Final: 最优尝试'''

'''(rn9+Cos+Maxdrop+erase)
nohup python router.py main --model_chosen=ResNet9 --scheduler='Cos' \
    --lr=0.1 --max_epoch=200 --Erase=True >> log.best1 2>&1 &
'''


'''(rn9+Cos+Maxdrop+erase)
nohup python router.py main --model_chosen=ResNet9_MaxDropout --scheduler='Cos'\
    --lr=0.1 --max_epoch=200 --Erase=True >> log.best2 2>&1 &
'''
 

'''(rn9+Cos+Maxdrop+erase+GELU)*****
nohup python router.py main --model_chosen=ResNet9_MaxDropout --scheduler='Cos'\
    --lr=0.1 --max_epoch=200 --Erase=True --activation='GELU' >> log.best3 2>&1 &
'''


'''(srn9+Cos+Maxdrop+erase+GELU)
nohup python router.py main --model_chosen=simp_ResNet9_MaxDropout --scheduler='Cos'\
    --lr=0.1 --max_epoch=200 --Erase=True --activation='GELU' >> log.best4 2>&1 &
'''


'''(srn9_k33+Cos+CyclicLR+erase)
nohup python router.py main --model_chosen=simp_ResNet9_k33 --scheduler=Cos \
    --lr=0.1 --max_epoch=200 --Erase=True --Erase=True >> log.best5 2>&1 &
'''


'''(srn9_k33+Cos+CyclicLR+erase+GELU)
nohup python router.py main --model_chosen=simp_ResNet9_k33 --scheduler=Cos \
    --lr=0.1 --max_epoch=200 --Erase=True --Erase=True --activation='GELU'>> log.best6 2>&1 &
'''


'''(srn9_k333+Cos+CyclicLR+erase+GELU)
nohup python router.py main --model_chosen=simp_ResNet9_k333 --scheduler=Cos \
    --lr=0.1 --max_epoch=200 --Erase=True --Erase=True --activation='GELU'>> log.best7 2>&1 &
'''