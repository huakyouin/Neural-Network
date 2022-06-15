## 项目导航

### 环境配置
```bash
conda create --name easy python=3.8 notebook
conda activate easy
## 按自己的配置修改cuda版本
conda install pytorch torchvision torchaudio cudatoolkit=11.3
pip install torch-ema wandb scipy
```

### Dataset
首先下载好oracle_fs.zip和oracle_source.zip,将它们放到与项目主文件夹平级的data文件夹中并解压
```bash
unzip oracle_fs.zip
unzip oracle_source.zip
```

### Train and Eval
详见项目文件夹中的`experiment.ipynb`文件。

