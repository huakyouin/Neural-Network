from pprint import pprint
import torch
# Default Configs for training

device = torch.device('cuda:0')

class Config:

    # 按批加载数据
    batchsize=128
    download=False  # 是否下载数据集
    datapath='default'
    Erase=False     # 训练数据是否局部擦除
    num_classes=10   # 数据集类别总数

    # 模式：train 或 test
    mode='train'

    # 训练参数
    max_epoch=20
    model_chosen='ResNet_original'
    activation='ReLU'
    _activationList=['ReLU','Mish','ELU','RReLU','GELU']
    activation_positive= activation in _activationList  # 是否有效激活
    show_net=False # 是否展示网络

    # 优化器参数
    opt='SGD'
    lr=0.01
    scheduler=None
    weight_decay = 5e-4

    # 损失函数
    loss='CrossEntropyLoss'
    _losslist=['CrossEntropyLoss','SmoothL1Loss','BCEWithLogitsLoss']
    loss_positive=loss in _losslist

    # visualization
    visualize=False    # 可视化开关
    log_name=None     # 自定义日志名
    # port = 8097
    plot_every_n_loop = 1  # vis every N loop


    # use_adam = False # Use Adam optimizer
    # use_chainer = False # try match everything as chainer
    # use_drop = False # use dropout in RoIHead

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)
        self.activation_positive= self.activation in self._activationList
        self.loss_positive=self.loss in self._losslist

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

args = Config()
