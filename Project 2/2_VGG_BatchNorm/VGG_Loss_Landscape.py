import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from loaders import get_cifar_loader,Seek_dir

# ## Constants (parameters) initialization

num_workers = 4

device= torch.device('cuda:0')

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')




# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)




# This function is used to calculate the accuracy of model classification
def get_accuracy(model):
    ## --------------------
    # Add code as needed
    size = 0
    correct = 0
    with torch.no_grad():
        for data in val_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            _, y_pred = torch.max(outputs.data, 1)
            size += y.size(0)
            correct += (y_pred == y).sum().item()

    print('Accuracy: %.2f %% ' % (100 * correct / size))

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    batches_n = len(train_loader)
    losses_list = []
    grads = []
    
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        grad = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve
        size = 0

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()
            ## --------------------
            # Add your code
            losses_list.append(loss.cpu().detach() )
            _, y_pred = torch.max(prediction.data, 1)
            learning_curve[epoch]+=(y_pred== y).sum().item()
            size += y.size(0)

            loss.backward()
            optimizer.step()
        learning_curve[epoch]/=size
    # Test your model and save figure here (not required)
    # remember to use model.eval()
    model.eval()
    get_accuracy(model)

    return losses_list,learning_curve



# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape(iteration,VGG_A_max_curve,
                        VGG_A_min_curve,VGG_A_BN_max_curve,
                        VGG_A_BN_min_curve):
    fig=plt.figure(0)
    plt.style.use('ggplot')
    # plot VGG_A curve
    plt.plot(iteration, VGG_A_max_curve, c='green')
    plt.plot(iteration, VGG_A_min_curve, c='green')
    plt.fill_between(iteration, VGG_A_max_curve, 
                     VGG_A_min_curve, color='lightgreen', 
                     label='Standard VGG')

    # plot VGG_A_BatchNorm  curve
    plt.plot(iteration, VGG_A_BN_max_curve, c='firebrick')
    plt.plot(iteration, VGG_A_BN_min_curve, c='firebrick')
    plt.fill_between(iteration, VGG_A_BN_max_curve, 
                     VGG_A_BN_min_curve, color='lightcoral',
                     label='Standard VGG + BatchNorm')
    
    # configs
    plt.xticks(np.arange(0, iteration[-1], 1000))
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss Landscape')
    plt.legend(loc='upper right', fontsize='x-large')
    savepath=Seek_dir('exp')+'/Loss_landscape_VGG_Cmp_BN.png'
    plt.savefig(savepath,dpi=300)
    plt.close(0)


def plot_acc_curve(iteration,VGG_A_acc,VGG_A_norm_acc):
    fig=plt.figure(0)
    plt.style.use('ggplot')
    plt.plot(iteration, VGG_A_acc, c='green',label='Standard VGG')
    plt.plot(iteration, VGG_A_norm_acc, c='firebrick',label='Standard VGG + BatchNorm')
     # configs
    plt.xticks(range(0, 22))
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuary')
    plt.title('Accuary Curve')
    plt.legend(loc='best', fontsize='x-large')
    savepath=Seek_dir('exp')+'/Train_Acc_VGG_Cmp_BN.png'
    plt.savefig(savepath,dpi=300)
    plt.close(0)


if __name__ == '__main__':
    lrs=[1e-4]
    # lrs=[2e-3, 1e-4,5e-4]
    batch_size = 128
    epoch_num=20
    set_random_seeds(seed_value=2022, device=device)
    
    VGG_A_losses = []
    VGG_A_BN_losses = []
    VGG_A_acc=[]
    VGG_A_bn_acc=[]
    
    for lr in lrs:
        model = VGG_A()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        a1,a2=train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epoch_num)
        VGG_A_losses.append(a1)
        VGG_A_acc.append(a2)
        
        model = VGG_A_BatchNorm()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        b1,b2=train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epoch_num)
        VGG_A_BN_losses.append(b1)
        VGG_A_bn_acc.append(b2)

   
    VGG_A_losses = np.array(VGG_A_losses)
    VGG_A_BN_losses = np.array(VGG_A_BN_losses)
    VGG_A_acc=np.array(VGG_A_acc)
    VGG_A_bn_acc=np.array(VGG_A_bn_acc)

    iteration = []
    VGG_A_min_curve = []
    VGG_A_max_curve = []
    VGG_A_BN_min_curve = []
    VGG_A_BN_max_curve = []
    
    VGG_A_min = VGG_A_losses.min(axis=0).astype(float)
    VGG_A_max = VGG_A_losses.max(axis=0).astype(float)
    VGG_A_BN_min = VGG_A_BN_losses.min(axis=0).astype(float)
    VGG_A_BN_max = VGG_A_BN_losses.max(axis=0).astype(float)
    for i in range(len(VGG_A_min)):
        if i%30 == 0:
            VGG_A_min_curve.append(VGG_A_min[i])
            VGG_A_max_curve.append(VGG_A_max[i])
            VGG_A_BN_min_curve.append(VGG_A_BN_min[i])
            VGG_A_BN_max_curve.append(VGG_A_BN_max[i])
            iteration.append(i)
    
    plot_acc_curve(range(1,21),VGG_A_acc[0],VGG_A_bn_acc[0])
    
    plot_loss_landscape(iteration,VGG_A_max_curve,
                        VGG_A_min_curve,VGG_A_BN_max_curve,
                        VGG_A_BN_min_curve)
    