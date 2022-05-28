import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blip import blip_decoder
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, coco_caption_eval,coco_caption_DCC_eval


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # evaluate
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10

    result = []
    for image, image_id in metric_logger.log_every(data_loader, print_freq, header): 
        image = image.to(device)       
        captions = model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], 
                                  min_length=config['min_length'])
        for caption, img_id in zip(captions, image_id):
            result.append({"image_id": img_id.item(), "caption": caption})
  
    return result


def main(args, config):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Model #### 
    print("Creating model..")
    if args.newtrain:
        print('new training!')
        selected=False
    elif args.usemyown is not None:
        print('use my model!')
        selected=args.usemyown
    elif args.pretrain:
        print('new training use pretrain!')
        selected=config['pretrained']
    else:
        print('use model in paper!')
        selected=config['paper_own']

    model = blip_decoder(pretrained=selected,
                            image_size=config['image_size'], vit=config['vit'],vit_grad_ckpt=config['vit_grad_ckpt'], 
                            vit_ckpt_layer=config['vit_ckpt_layer'], 
                            prompt=config['prompt'])
    print('model created!')
    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
            
    for idx in range(len(args.testfile)):
        #### Dataset #### 
        print("Creating captioning dataset")
        if args.valfile!='': vf=args.valfile
        tf=args.testfile[idx]
        train_dataset, val_dataset, test_dataset = create_dataset('caption_coco_DCC', config,tf=tf)  

        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()            
            samplers = create_sampler([train_dataset,val_dataset,test_dataset], [True,False,False], num_tasks, global_rank)         
        else:
            samplers = [None, None, None]
        
        train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                            batch_size=[args.batch_size]*3,num_workers=[4,4,4],
                                                            is_trains=[True, False, False], collate_fns=[None,None,None])  
               
        # 预测
        if args.valfile!='':
            print('Start evaluating ',vf,'..')
            val_result = evaluate(model_without_ddp, val_loader, device, config)  
            val_result_file = save_result(test_result, args.result_dir, 'val_'+args.pred_save_name[idx], remove_duplicate='image_id')  
            if utils.is_main_process():
                coco_val = coco_caption_DCC_eval(config['ann_root_DCC'],val_result_file,'val',vf=vf,tf=tf)
                log_stats = {**{f'val_{k}': v for k, v in coco_val.eval.items()},}

                with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                    f.write('\n'+tf+':\n')
                    f.write(json.dumps(log_stats) + "\n") 

        print('Start evaluating ',tf,'..')
        test_result = evaluate(model_without_ddp, test_loader, device, config)  
        test_result_file = save_result(test_result, args.result_dir, args.pred_save_name[idx], remove_duplicate='image_id')    
        if utils.is_main_process():
            coco_test = coco_caption_DCC_eval(config['ann_root_DCC'],test_result_file,'test',tf=tf)         
            log_stats = {**{f'test_{k}': v for k, v in coco_test.eval.items()},}
            global df
            df[str(args.objects[idx])]=['%.1f'%(v*100) for k, v in coco_test.eval.items()]
            df.index = [k for k, v in coco_test.eval.items()]
            with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                f.write('\n'+tf+':\n')
                f.write(json.dumps(log_stats) + "\n")                   


def eval_f1(keyword,pred,gt):
    tp=0;fp=0;fn=0
    df1=pd.DataFrame(pred)
    df2=pd.DataFrame(gt)
    for index in range(len(df1)):
        id=df1.loc[index].image_id
        pred_cap=df1.loc[index].caption
        pred_positive=True if pred_cap.find(keyword)!= -1 else False
        temp=df2[df2.image_id==id]
        len1=len(temp)
        temp=temp[temp.caption.str.contains(keyword)]
        if pred_positive and len(temp)==len1: tp+=1
        if pred_positive and len(temp)!=len1: fp+=1
        if not pred_positive and len(temp)==len1: fn+=1
    
    print('-'*5,'keyword:',keyword,'-'*5)
    print('tp: %d, fp: %d, fn: %d, F1_score: %.3f'%(tp,fp,fn,tp/(tp+0.5*fp+0.5*fn)))
    print('-'*25) 
    return '%.2f'%(tp/(tp+0.5*fp+0.5*fn)*100)

##############################  不太用改的设置  #########################
parser = argparse.ArgumentParser()    
args = parser.parse_args(args=[])
args.evaluate=True
args.device='cuda'  
args.newtrain=False
args.pretrain=False
args.seed=42
args.world_size=1
args.distributed=True
args.dist_url='env://eval'

############################  可能需要修改的设置  #######################
args.config='./configs/caption_coco.yaml'
args.output_dir='output/DCC_eval'
args.usemyown='output/Caption_coco_DCC_pretrain_1/checkpoint_best.pth'
args.batch_size=64

args.objects=['bottle','bus','couch','microwave','pizza','racket','suitcase','zebra']
# args.objects=['bottle']
args.pred_save_name=[str(item)+'_predict' for item in args.objects]
args.valfile=''
args.testfile=['captions_split_set_%s_val_test_novel2014.json'%(item) for item in args.objects]
args.output_dir='output/DCC_eval'
args.result_dir='output/DCC_eval/result'

##############################  不太用改的设置  #########################
config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
Path(args.result_dir).mkdir(parents=True, exist_ok=True)
yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w')) 

df = pd.DataFrame()
main(args, config)

print('markdown格式结果：')
print(df.to_markdown())
print('html格式结果：')
print(df.to_html())


## 下面计算F1得分

from pycocotools.coco import COCO
############################  可能需要修改的设置  #######################
ann_file="annotation/DCC/captions_val_test2014.json" 
##############################  不太用改的设置  #########################
# ground_truth
annFile= ann_file
coco = COCO(annFile)
gt = coco.dataset['annotations']
group1=args.objects
f1=[]
for name in group1:
    pred_file='output/DCC_eval/result/%s_predict.json' %name
    # prediction
    with open(pred_file, 'rb') as f:
        pred = json.load(f)
    f1.append(eval_f1(name,pred,gt))
df3 = pd.DataFrame([f1])
df3.index=['F1']
df3.columns=group1
df=pd.concat([df,df3])

print(df3)
print('markdown格式结果：')
print(df.to_markdown())
print('html格式结果：')
print(df.to_html())
df.to_markdown(args.output_dir+'/final_result.md')
df.to_html(args.output_dir+'/final_result.html')