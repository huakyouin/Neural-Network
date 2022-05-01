"""
Data loaders
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.datasets as datasets
import os
from pathlib import Path

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

class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self):
        return self.dataset.__getitem__()

    def __len__(self):
        return min(self.n_items, len(self.dataset))


def get_cifar_loader(root=Seek_dir('JXHdatasets',4), batch_size=128, train=True, shuffle=True, num_workers=4, n_items=-1):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    data_transforms = transforms.Compose(
        [transforms.ToTensor(),
        normalize])

    dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=data_transforms)
    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader

if __name__ == '__main__':
    train_loader = get_cifar_loader()
    for X, y in train_loader:
        print(X[0])
        print(y[0])
        print(X[0].shape)
        img = np.transpose(X[0], [1,2,0])
        plt.imshow(img*0.5 + 0.5)
        # plt.savefig('sample.png')
        plt.show()
        print(X[0].max())
        print(X[0].min())
        break