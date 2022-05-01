from pathlib import Path
import torchvision
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset

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


def loading(name,dir=None,BATCH=512):
    defaultPath=Seek_dir() if dir is None else dir
    if name in ['mnist','MNIST']:
        train_dataset = torchvision.datasets.MNIST(root=defaultPath, train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = torchvision.datasets.MNIST(root=defaultPath, train=False, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH, shuffle=False)
        model_dir = Seek_dir('MNIST_model_data')
        return train_dataset,test_dataset,train_loader,test_loader,model_dir
    elif name in ['cifar','CIFAR']:
        transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])
        train_dataset = torchvision.datasets.CIFAR10(root=defaultPath, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=defaultPath, train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH, shuffle=False, num_workers=4)
        model_dir = Seek_dir('cifar10_model_data')
        return train_dataset,test_dataset,train_loader,test_loader,model_dir


if __name__ == '__main__':
    _,_,train_loader,_,_ = loading(name='cifar',BATCH=128)
    for X, y in train_loader:
        print(X[0])
        print(y[0])
        print(X[0].shape)
        img = np.transpose(X[0], [1,2,0])
        # plt.imshow(img*0.5 + 0.5)
        # plt.savefig('sample.png')
        # plt.show()
        print(X[0].max())
        print(X[0].min())
        break