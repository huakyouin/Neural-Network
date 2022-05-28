## 环境配置

数据集放置如下

```bash
|-- coco
|   |-- train2014
|   |-- val2014
|-- ntt
```

首先从网盘下载[data](https://www.123pan.com/s/Cz8DVv-b0wkv), [glove.6B](https://www.123pan.com/s/Cz8DVv-Z0wkv), [tools](https://www.123pan.com/s/Cz8DVv-c0wkv), 放置在项目主文件夹下；
```bash
# 部署文件
cd ntt
unzip data
unzip tools
unzip glove.6B -d .vector_cache

# 底层环境部署 如果服务器配置齐全可以跳过
apt-get update && \
    apt-get install -y \
    ant \
    vim \
    ca-certificates-java \
    nano \
    openjdk-8-jdk \
    unzip \
    wget && \
    apt-get clean
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 >> /etc/profile
source /etc/profile
update-ca-certificates -f && export JAVA_HOME

# 部署conda环境
conda env create -f environment.yml
conda activate ntt
pip install tensorflow==1.0.0 pyyaml
pip install bert_embedding -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 训练&评估
从头训练
```bash
python -u main.py --path_opt cfgs/noc_coco_res101.yml --batch_size 16 --cuda True --num_workers 10 --max_epoch 31 --glove_6B_300 True --att_model newtopdown --val_split val 
```
源码存在一定问题，训练完的模型无法使用，因此在训练代码上直接加上了预测，就不提供模型了如需复现请重跑，但是预测文件保存在save文件夹中，后续评估仍然可以直接运行
- `--cuda`: bool| use cuda
- `--mGPUs` bool| use GPUs>=1
- `--start_from`: str| pretrained model path
- `--load_best_score`: number| =1 is load best model; else load last model
- `--val_split`: str| val or test
- `--val_every_epoch`: int| val model every (int) epoch


**Note: 下面的流程无需且无法用上面的环境，更换一个含有pandas,tabulate和pycocotools,tensorboard模块的环境**


## 小noc目标的评估
调整`Eval.py`文件中的路径并运行：
```bash
python Eval.py
python F1score.py
```


### 我的结果


- 整个noc测试集
<div>
<table align="center" >
<tr>
<th style='text-align:center;' colspan=4>Metrics</th>
</tr>
<tr>
<td style='text-align:center;'>Bleu_4</td>
<td style='text-align:center;'>METEOR</td>
<td style='text-align:center;'>CIDEr</td>
<td style='text-align:center;'>SPICE</td>
</tr>
<tr>
<td style='text-align:center;'>30.6</td>
<td style='text-align:center;'>24.9</td>
<td style='text-align:center;'>93.4</td>
<td style='text-align:center;'>18.2</td>
</tr>
</table>
</div>


- 小noc目标的测试集

|            |  bottle   |   bus    |  couch   | microwave | pizza |  racket  | suitcase | zebra |
| :----: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| **BlEU4** |     28.9 |  22.1 |    33.8 |    **34.4** |    27.5 |     22.2 |       23.5 |    23.1 |
| **CIDEr** | **83.3** |  45.9 |    68   |        61.3 |    52.1 |     27.8 |       55.6 |    36.7 |
| **METEOR** |     22.9 |  20.8 | **25.7** |        25   |    21.8 |     23.6 |       20.3 |    22.9 |
| **SPICE** |     16.1 |  14.6 | **18.3** |        16.2 |    16.4 |     14.8 |       13.1 |    16.6 |
| **F1** | 20.0 | 63.7 | 21.3 | 36.6 | 41.9 | 9.8  | 9.1  | **72.8** |
