# CIFAR-10实验命令记录

> Note: 如需复现请将工作区加载到router.py所在路径, 然后命令行输入以下代码

> Note: 本实验除了pytorch外还需要安装tqdm,fire,matplotlib模块

# Stage 1: try different models

**ResNet18**

`python router.py main --model_chosen=ResNet18`

Accuracy of the network on train set: 92.62 %

Accuracy of the network on test set: 87.84 %

Finished Training! Totally Training Time Cost 1518.0233380794525



**DLA**

`python router.py main --model_chosen=DLA`

Accuracy of the network on train set: 92.83 %

Accuracy of the network on test set: 88.01 %

Finished Training! Totally Training Time Cost 2926.0044569969177



**ResNet9**

`python router.py main --model_chosen=ResNet9`

Accuracy of the network on train set: 93.11 %

Accuracy of the network on test set: 86.98 %

Finished Training! Totally Training Time Cost 373.71886682510376 



**SimpResNet9**

`python router.py main --model_chosen=simp_ResNet9`

Accuracy of the network on train set: 90.63 %

Accuracy of the network on test set: 85.06 %

Finished Training! Totally Training Time Cost 370.20921540260315 





# Stage 2: learning rate schedulers

**CyclicLR**

`python router.py main --model_chosen=simp_ResNet9 --scheduler='CyclicLR'`



Accuracy of the network on train set: 95.21 %

Accuracy of the network on test set: 90.28 %

Finished Training! Totally Training Time Cost 373.96056294441223 



**OneCycleLR**

`python router.py main --model_chosen=simp_ResNet9 --scheduler='OneCycleLR'`



Accuracy of the network on train set: 94.21 %

Accuracy of the network on test set: 89.50 %

Finished Training! Totally Training Time Cost 375.6388645172119 



**CosLR**

`python router.py main --model_chosen=simp_ResNet9 --scheduler='Cos'`



Finished Training! Totally Training Time Cost 373.96056294441223 

Accuracy of the network on train set: 94.15 %

Accuracy of the network on test set: 90.24 %



# Stage 3: activations


**Mish**

`python router.py main --model_chosen=simp_ResNet9 --scheduler='CyclicLR' --activation='Mish'`



Accuracy of the network on train set: 93.41 %

Accuracy of the network on test set: 88.91 %

Finished Training! Totally Training Time Cost 352.5207483768463 



**ELU**

`python router.py main --model_chosen=simp_ResNet9 --scheduler='CyclicLR' --activation='ELU'`



Accuracy of the network on train set: 89.69 %

Accuracy of the network on test set: 85.96 %

Finished Training! Totally Training Time Cost 483.8410952091217 



**RReLU**

`python router.py main --model_chosen=simp_ResNet9 --scheduler='CyclicLR' --activation='RReLU'`



Accuracy of the network on train set: 30.70 %

Accuracy of the network on test set: 31.30 %

Finished Training! Totally Training Time Cost 266.79511761665344  



**GELU**

`python router.py main --model_chosen=simp_ResNet9 --scheduler='CyclicLR' --activation='GELU'`


Accuracy of the network on train set: 95.09 %

Accuracy of the network on test set: 90.12 %

Finished Training! Totally Training Time Cost 417.02611684799194 





# Stage 4: kernel size

先前ResNet9和simpResNet9已经对深层卷积块改变进行了探索，这一阶段只对第一卷积块进行实验

**SimpResNet9--Firstblock[3,5]**

`python router.py main --model_chosen=simp_ResNet9`



Accuracy of the network on train set: 90.63 %

Accuracy of the network on test set: 85.06 %

Finished Training! Totally Training Time Cost 370.20921540260315 



**SimpResNet9--Firstblock[7]**

`python router.py main --model_chosen=simp_ResNet9_k7`



Accuracy of the network on train set: 87.87 %

Accuracy of the network on test set: 82.34 %

Finished Training! Totally Training Time Cost 213.83135890960693 



**SimpResNet9--Firstblock[3,3,3]**

`python router.py main --model_chosen=simp_ResNet9_k333`



Accuracy of the network on train set: 91.52 %

Accuracy of the network on test set: 86.68 %

Finished Training! Totally Training Time Cost 306.2286319732666 



**SimpResNet9--Firstblock[3,3]**

`python router.py main --model_chosen=simp_ResNet9_k33`



Accuracy of the network on train set: 89.81 %

Accuracy of the network on test set: 84.91 %

Finished Training! Totally Training Time Cost 261.62371039390564 



**SimpResNet9--Firstblock[3,3]+CyclicLR**

`python router.py main --model_chosen=simp_ResNet9_k33 --scheduler='CyclicLR'`



Accuracy of the network on train set: 95.12 %

Accuracy of the network on test set: 90.19 %

Finished Training! Totally Training Time Cost 261.08778858184814 

相比首块核=[3,5] 基本差不多


# Stage 5: loss function

Note: default loss is CrossEntropyLoss  acc = 85 %


**SmoothL1Loss**

`python router.py main --model_chosen=simp_ResNet9 --loss=SmoothL1Loss`

Accuracy of the network on train set: 61.59 %

Accuracy of the network on test set: 61.06 %

Finished Training! Totally Training Time Cost 1103.1858491897583 



**BCEWithLogitsLoss**

`python router.py main --model_chosen=simp_ResNet9 --loss=BCEWithLogitsLoss`

Accuracy of the network on train set: 82.23 %

Accuracy of the network on test set: 79.86 %

Finished Training! Totally Training Time Cost 1130.8084375858307 



# Final: 最优尝试



**rn9+CosLR+erase**

```bash
nohup python router.py main --model_chosen=ResNet9 --scheduler='Cos' \
  --lr=0.1 --max_epoch=200 --Erase=True >> best1.log 2>&1 &
```

Accuracy of the network on train set: 98.31 %

Accuracy of the network on test set: 95.21 %

Finished Training! Totally Training Time Cost 5772.65283370018 



**rn9+CosLR+Maxdropout+erase**

```bash
nohup python router.py main --model_chosen=ResNet9_MaxDropout --scheduler='Cos' \
  --lr=0.1 --max_epoch=200 --Erase=True >> best2.log 2>&1 &
```

Accuracy of the network on train set: 98.39 %

Accuracy of the network on test set: 95.21 %

Finished Training! Totally Training Time Cost 10016.914810419083 



**rn9+CosLR+Maxdropout+erase+GELU**

```bash
nohup python router.py main --model_chosen=ResNet9_MaxDropout --scheduler='Cos' \
  --lr=0.1 --max_epoch=200 --Erase=True --activation='GELU' >> best3.log 2>&1 &
```

Accuracy of the network on train set: 98.45 %

Accuracy of the network on test set: 95.42 %

Finished Training! Totally Training Time Cost 9713.020115613937 



**srn9+CosLR+Maxdropout+erase+GELU**

```bash
nohup python router.py main --model_chosen=simp_ResNet9_MaxDropout --scheduler='Cos' \
  --lr=0.1 --max_epoch=200 --Erase=True --activation='GELU' >> best4.log 2>&1 &
```

Accuracy of the network on train set: 96.69 %

Accuracy of the network on test set: 94.03 %

Finished Training! Totally Training Time Cost 9105.754658937454 



**srn9_k33+CosLR+erase**

```bash
nohup python router.py main --model_chosen=simp_ResNet9_k33 --scheduler=Cos \
  --lr=0.1 --max_epoch=200 --Erase=True >> best5.log 2>&1 &
```

Accuracy of the network on train set: 96.42 %

Accuracy of the network on test set: 93.99 %

Finished Training! Totally Training Time Cost 4635.8597095012665  



**srn9_k33+CosLR+erase+GELU**

```bash
nohup python router.py main --model_chosen=simp_ResNet9_k33 --scheduler=Cos \
  --lr=0.1 --max_epoch=200 --Erase=True --activation='GELU'>> best6.log 2>&1 &
```

Accuracy of the network on train set: 96.86 %

Accuracy of the network on test set: 94.14 %

Finished Training! Totally Training Time Cost 4479.505462169647 



**srn9_k333+CosLR+erase+GELU**

```bash
nohup python router.py main --model_chosen=simp_ResNet9_k333 --scheduler=Cos \
  --lr=0.1 --max_epoch=200 --Erase=True --activation='GELU'>> best7.log 2>&1 &
```
Accuracy of the network on train set: 97.45 %

Accuracy of the network on test set: 94.80 %

Finished Training! Totally Training Time Cost 10692.597094297409 


**srn9_k35+CosLR+erase+GELU**

```bash
nohup python router.py main --model_chosen=simp_ResNet9 --scheduler='Cos' \
  --lr=0.1 --max_epoch=200 --Erase=True --activation='GELU' >> best8.log 2>&1 &
```

Accuracy of the network on train set: 97.01 %

Accuracy of the network on test set: 94.05 %

Finished Training! Totally Training Time Cost 4915.570363283157  
