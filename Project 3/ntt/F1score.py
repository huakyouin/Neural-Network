from pycocotools.coco import COCO
import json
from json import encoder
import pandas as pd
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

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

            
pred_file='save/noc_coco_1024_adam/preds.json'
ann_file='tools/coco-caption/annotations/captions_val2014.json' 
# prediction
with open(pred_file, 'rb') as f:
    pred = json.load(f)
# ground_truth
annFile= ann_file
coco = COCO(annFile)
gt = coco.dataset['annotations']
group1=['bottle','bus','couch','microwave','pizza','racket','suitcase','zebra']
for name in group1:
    eval_f1(name,pred,gt)