import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import sys
#from build_vocab import Vocabulary
#from pycocotools.coco import COCO
import glob, os
import ntpath
from torch.autograd import Variable 
import torchvision
import random
import tqdm
import json

dir = '/home/zhangr/data/GTEx_Tiles_jpg/'      
ids = glob.glob(dir + "/*/*.jpg")
print( "ids image number: " + str(len(ids)))
distance = 1500

mask_id = []
with open('train_enc_imgs_lt.json', 'r') as fin:
    mask_id = json.load(fin)
with open('val_enc_imgs_lt.json', 'r') as fin:
    mask_id += json.load(fin)
with open('test_enc_imgs_lt.json', 'r') as fin:
    mask_id += json.load(fin)

mask_id_tmp = [slide_id[:-4] for slide_id in mask_id] 
mask_id = set(mask_id_tmp)

print(f'mask id: {len(mask_id)}')

ids_tmp = []
for path in ids:
    basename = os.path.basename(path)
    array = basename.split('_')
    slide_id = array[0]
    if slide_id in mask_id:
        ids_tmp.append(path)
ids = ids_tmp

print(f'tile num after filter: {len(ids)}')

slide_imgs = {}
for path in ids:
    basename = os.path.basename(path)
    array = basename.split('_')
    slide_id = array[0]
    if slide_id in slide_imgs:
        slide_imgs[slide_id].append(path)
    else:
        slide_imgs[slide_id] = [path]

print(f'tile numbers: {len(ids)}')
print(f'slide numbers: {len(slide_imgs)}')


def gen_adj(sub_ids):
    for i in range(len(sub_ids)):
        path = sub_ids[i]
        basename = os.path.basename(path)
        basename = basename[:-4]
        array = basename.split('_')
        slide_id = array[0]
        x = int(array[1][4:])
        y = int(array[2][4:])
        
        if os.path.exists('adj_img/' + basename + '.txt'):
            continue

        adj_imgs = []
        
        for can_imgs in slide_imgs[slide_id]:
            can_basename = os.path.basename(can_imgs)
            can_basename = can_basename[:-4]
            can_array = can_basename.split('_')
            #mag = can_array[3]
            mag = 'mag20'
            can_x = int(can_array[1][4:])
            can_y = int(can_array[2][4:])
            if can_x == x and can_y == y:
                continue
            
            if mag == 'mag20':
                if abs(x - can_x) <= distance and abs(y-can_y) <= distance:
                    adj_imgs.append(can_imgs)
            else:
                assert mag == 'mag40'
                if abs(x - can_x) <= 2*distance and abs(y-can_y) <= 2*distance:
                    adj_imgs.append(can_imgs)

        fout = open('adj_img/' + basename + '.txt', 'w')
        for can_path in adj_imgs:
            fout.write(can_path + '\n')
        fout.close()

from multiprocessing import Process

proc_num = 37
inter = int(len(ids) / proc_num) + 1
p_lt = []

for i in range(proc_num):
    start = i*inter
    end = (i+1)*inter
    if end > len(ids):
        end = len(ids)
    sub_ids = ids[start: end]
    p_lt.append(Process(target=gen_adj, args=(sub_ids,)))
    p_lt[i].start()

for i in range(proc_num):
    p_lt[i].join()



