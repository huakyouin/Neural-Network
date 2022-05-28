## 项目使用指南

如需验证我的模型，请在[网盘](https://www.123pan.com/s/Cz8DVv-L0wkv)上下载output.zip放到主文件夹内并解压，并确认`MyEval.py`中的路径设置正确

### step 1：环境配置

基础环境创建：

```bash
conda create --name blip python=3.8 notebook
conda activate blip
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3
# 安装 timm、transformers、fairscale、pycocoevalcap、opencv-python、ruamel.yaml等依赖
pip install timm==0.4.12 transformers==4.15.0 fairscale==0.4.4 pycocoevalcap opencv-python ruamel.yaml nltk pandas tabulate
```

pycocoevalcap包在任务中需要额外下载斯坦福nlp包中的'stanford-corenlp-3.6.0.jar'和'stanford-corenlp-3.6.0-models.jar'，运行中下载很慢，因此建议手动下载：
1. 从[网盘](https://www.123pan.com/s/Cz8DVv-e8wkv)下载两个.jar文件
2. 命令行输入`pip show pycocoevalcap`找到包pycocoevalcap位置并进入
3. 在../pycocoevalcap/spice/lib/路径下放置两个文件

### step 2：准备COCO数据集与annotation

1. 首先下载好coco2014数据集，在本项目中images文件夹中应包含test2014、train2014和val2014，结构如下：

```bash
|--- BLIP-main
|--- coco
|       |--- images
|               |--- train2014
|               |--- val2014
|               |--- test2014
```

2. 修改configs/caption_coco.yaml中的image_root，参考image_root='/home/newdisk/jxh/课程项目/神网PJ_3/coco/images/'

3. 从[网盘](https://www.123pan.com/s/Cz8DVv-z0wkv)下载一整个annotation.zip，若放在主文件夹内解压可以跳过第4步

4. 修改configs/caption_coco.yaml中的ann_root、coco_gt_root、ann_root_DCC、ann_root_mine


### step 3：在COCO(Hendricks划分)上训练
分布式训练在bash上即使nohup也不能关闭终端，使用tmux作为终端取代nohup更好

1. 随机初始化进行训练
```tmux
CUDA_VISIBLE_DEVICES=1,2,3 \
python -m torch.distributed.run --nproc_per_node=3 train_caption_DCC.py \
--newtrain
```
- `--newtrain`: 随机初始化
- `--nproc_per_node`: GPU数量

2. 预训练+微调
在`configs/caption_coco.yaml`修改训练超参数，并运行：
```tmux
CUDA_VISIBLE_DEVICES=3,2,1,0 \
python -m torch.distributed.run --nproc_per_node=4 train_caption_DCC.py \
--pretrain --output_dir output/Caption_coco_DCC_pretrain_1
```
- `--pretrain`: 使用上游预训练模型
- `--output_dir`: 输出日志位置


3. 恢复训练或继续训练

先在`configs/caption_coco.yaml`调小学习率`init_lr`等超参数, 并运行：

中然后命令行输入：
```tmux
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run --nproc_per_node=4 train_caption_DCC.py \
--usemyown 'output/Caption_coco_DCC_train/checkpoint_best.pth'
```
- `usemywon`: 之前的训练模型路径

### step 4：评估自训练模型

单个测试划分上的示例：
```bash
python -m torch.distributed.run --nproc_per_node=4 train_caption_DCC.py --evaluate \
--usemyown 'output/Caption_coco_DCC_pretrain/checkpoint_best.pth' \
--testfile 'captions_val_test2014.json' \
--valfile 'captions_val_val2014.json' \
--output_dir 'output/DCC_eval'
```
- `testfile`: 测试集划分文件名


由于测试划分较多，我封装了评估模块,如遇问题请进入`MyEval.py`修改:
```bash
python -m torch.distributed.run --nproc_per_node=4 MyEval.py
```

### 【Extra】源码模型评测
#### 源码模型在COCO(karpathy划分)上评测
```bash
python -m torch.distributed.run --nproc_per_node=1 train_caption.py --evaluate
```
--nproc_per_node: 使用的GPU数
--evaluate: 是否使用评估模式

#### 源码模型在COCO(Hendricks划分)上评测
```bash
python -m torch.distributed.run --nproc_per_node=1 train_caption_DCC.py --evaluate
```

### Acknowledgement
The implementation of BLIP relies on resources from <a href="https://github.com/salesforce/ALBEF">ALBEF</a>, <a href="https://github.com/huggingface/transformers">Huggingface Transformers</a>, and <a href="https://github.com/rwightman/pytorch-image-models/tree/master/timm">timm</a>. We thank the original authors for their open-sourcing.