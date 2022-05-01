import matplotlib.pyplot as plt 
from loaders import Seek_dir

exp_dir =  Seek_dir('exp')


''' 以下plot为LeNet中用到的'''
def plot_test_train_acc(train_accs,test_accs,title,exp_path,ylim=[0,1]):
    plt.figure(figsize=(5,5),dpi=1000)
    plt.clf()
    plt.style.use('ggplot')
    plt.plot(train_accs, label='train')
    plt.plot(test_accs, label='test')
    plt.xlabel('epoch')
    plt.ylim(ylim)
    plt.ylabel('accuracy')
    plt.title(title)
    plt.legend()
    plt.savefig(exp_path)
    plt.close()

def plot_weight_conv3(weight,title,savename):
    plt.figure(figsize=(5,5),dpi=1000)
    plt.clf()
    plt.imshow(weight,cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(exp_dir+'/'+savename+'.png')


def compare_weight_whether_prune(before_weight,after_weight):
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(before_weight,cmap='gray')
    plt.title('before pruning')
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(after_weight,cmap='gray')
    plt.axis('off')
    plt.title('after pruning')
    plt.savefig(exp_dir+'/conv3_weight_compare.png')


''' 以下plot为ResNet中用到的'''
def compare_accs(tra1,ta1,tra2,ta2,title1,title2,exp_path,ylim=[0,1]):
    plt.figure(figsize=(10,5),dpi=1000)
    plt.clf()
    plt.subplots_adjust(wspace=0.25)
    plt.style.use('ggplot')

    plt.subplot(1,2,1)
    plt.plot(tra1, label='Train Accuary')
    plt.plot(ta1, label='Test Accuary')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(ylim)
    plt.title(title1)
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(tra2, label='Train Accuary')
    plt.plot(ta2, label='Test Accuary')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(ylim)
    plt.title(title2)
    plt.legend()

    plt.savefig(exp_path)
    plt.close()

def plot_compare_weight_conv3(weight1,title1,weight2,title2,savename):
    plt.figure(figsize=(10,5),dpi=1000)
    plt.clf()

    plt.subplot(1,2,1)
    plt.imshow(weight1,cmap='gray')
    plt.title(title1)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(weight2,cmap='gray')
    plt.title(title2)
    plt.axis('off')

    plt.savefig(exp_dir+'/'+savename+'.png')


def cmp_loss_and_acc(loss1,loss2,
                    acc1,acc2,
                    model1,model2,
                    savename='cmp_loss_and_acc'):
    plt.figure(figsize=(10, 4),dpi=1000)
    plt.style.use('ggplot')
    # plt.rcParams['figure.figsize'] = (10.0, 4.0)
    plt.clf()
    plt.subplots_adjust(wspace=0.25)
    plt.subplot(1,2,1)
    plt.plot(loss1,label=model1)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(loss2,label=model2)
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(acc1,label=model1)
    plt.ylabel('Test Accuary')
    plt.xlabel('epoch')
    plt.plot(acc2,label=model2)
    plt.legend()

    plt.savefig(exp_dir+'/'+savename)
    plt.close()


