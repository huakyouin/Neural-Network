import numpy as np
import time
import os
import pickle
import opts
import pdb
import json
from misc import utils
import pandas as pd

df=pd.DataFrame()
noc_object = ['bottle','bus',  'couch', 'microwave', 'pizza', 'racket', 'suitcase', 'zebra']
dataset= ['data/DCC/captions_split_set_%s_val_test_novel2014.json'%item for item in noc_object]

pred_file='save/noc_coco_1024_adam/preds.json'
# prediction
with open(pred_file, 'rb') as f:
    pred = json.load(f)

for idx in range(len(dataset)):
    lang_stats = utils.language_eval(dataset[idx], pred, noc_object[idx], 'test', '')
    df[str(noc_object[idx])]=['%.1f'%(v*100) for k, v in lang_stats.items()]
    df.index = [k for k, v in lang_stats.items()]

print(df.to_markdown())