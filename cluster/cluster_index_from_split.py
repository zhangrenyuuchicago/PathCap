import json
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from shutil import copy
import os

cluster_num = 1

slide_id_img_num = {}
with open('../AE_triplet_loss/gen_rep_II/representation_all_3epoch.txt') as fin:  
    for line in fin:
        array = line.strip().split()
        img_name = array[0]
        slide_id = img_name.split('_')[0]
        if slide_id in slide_id_img_num:
            slide_id_img_num[slide_id] += 1
        else:
            slide_id_img_num[slide_id] = 1


base_name_lt = list(slide_id_img_num.keys())
print(f'slide num: {len(base_name_lt)}')

def get_cluster(index):
    fix_base = base_name_lt[index]
    img_rep = {}
    with open('./split/' + fix_base + '.txt') as fin:  
        for line in fin:
            array = line.strip().split()
            img_name = array[0]
            slide_id = img_name.split('_')[0]
            if fix_base == slide_id:
                rep = [float(array[i]) for i in range(1, len(array))]
                img_rep[img_name] = rep
    
    X = []
    img_name_lt = []

    for img_name in img_rep:
        img_name_lt.append(img_name)
        X.append(img_rep[img_name])
    
    if len(X) <= 20:
        return {}

    X = np.array(X)
    #print(X.shape)
    #X_embedded = TSNE(n_components=2, perplexity=50.0).fit_transform(X)

    kmeans = KMeans(n_clusters=cluster_num).fit(X)
    label_lt = list(kmeans.labels_)
    #print(label_lt)
    
    assert len(img_name_lt) == len(label_lt)
    cluster_img = {}
    for i in range(len(label_lt)):
        cluster_name = str(label_lt[i])
        if cluster_name in cluster_img:
            cluster_img[cluster_name].append(img_name_lt[i])
        else:
            cluster_img[cluster_name] = [img_name_lt[i]]
    
    return cluster_img

import json

def proc(index_lt):
    print(f'start proc {index_lt}')
    for base_index in index_lt:
        basename = base_name_lt[base_index]
        json_file_name = f'train_index_{cluster_num}/' + basename + '_slide_tile_cluster.json'
        if os.path.exists(json_file_name):
            continue
        print(f'process basename: {basename}')
        cluster_img = get_cluster(base_index)
        #base_cluster_img[basename] = cluster_img
        if len(cluster_img) == 0:
            continue

        with open(json_file_name, 'w') as outfile:
            json.dump(cluster_img, outfile)

from multiprocessing import Process

proc_num = 30
inter = int(len(base_name_lt) / proc_num)
p_lt = []

print(f'base name lt num: {len(base_name_lt)}')
print(f'base name set num: {len(set(base_name_lt))}')

for i in range(proc_num):
    index_lt = range(i*inter, (i+1)*inter)    
    print(f'process:{i}')
    print(index_lt)
    if i == (proc_num - 1):
        index_lt = range(i*inter, len(base_name_lt))
    p_lt.append(Process(target=proc, args=(index_lt,)))
    p_lt[i].start()

for i in range(proc_num):
    p_lt[i].join()

'''
base_cluster_img = {}
for base_index in range(len(base_name_lt)):
    basename = base_name_lt[base_index]
    print(f'process basename: {basename}')
    cluster_img = get_cluster(base_index)
    base_cluster_img[basename] = cluster_img
    break

import json
with open('slide_tile_cluster.json', 'w') as outfile:
    json.dump(base_cluster_img, outfile)
'''

