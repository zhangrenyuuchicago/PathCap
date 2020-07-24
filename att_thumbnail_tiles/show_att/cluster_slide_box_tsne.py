import json
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from shutil import copy
import os
import sys

slide_file = sys.argv[1]
cluster_folder = sys.argv[2]
pred_folder = sys.argv[3]

basename = os.path.basename(slide_file)
array = basename.split('.')
slide_id = array[0]

cluster_file_name = cluster_folder + '/' + slide_id + '_slide_tile_cluster.json'
with open(cluster_file_name, 'r') as fin:
    cluster = json.load(fin)

pred_file = pred_folder + '/' + slide_id + '.txt'

att_lt = []
hyp = ''
ref = ''

fin = open(pred_file, 'r')
while True:
    line = fin.readline().strip()
    if not line:
        break
    if line[0:2] == '[[':
        line = line[2:-2]
        array = line.split(',')
        att = [float(array[i]) for i in range(len(array))]
        att_lt.append(att)
    elif line == 'hyposthese:':
        line = fin.readline().strip()
        hyp = line
    elif line == 'reference:':
        line = fin.readline().strip()
        ref = line
fin.close()

assert len(att_lt) == len(hyp.split()) + 1
# normalize

for i in range(len(att_lt)):
    att = att_lt[i]
    max_w = max(att)
    min_w = min(att)
    if max_w > min_w:
        att = [(att[j] - min_w) / (max_w - min_w) for j in range(len(att))]
    else:
        att = [0.0 for j in range(len(att))]
    att_lt[i] = att

import openslide
import sys
from PIL import Image, ImageDraw

if not os.path.exists(slide_file):
    print('can not find svs;\nexit;')
    sys.exit()

slide = openslide.open_slide(slide_file)
level = slide.level_count - 1
width, height = slide.level_dimensions[level]
image = slide.read_region((0, 0), level, (width, height))
top_width, top_height = slide.level_dimensions[0]
thb_width = 1000
grid_size = 1000
if slide.properties['openslide.objective-power'] == '40':
    size = (2*grid_size, 2*grid_size)
elif slide.properties['openslide.objective-power'] == '20':
    size = (grid_size, grid_size)
else:
    print('system fault ')
    sys.exit()

rate = top_width * 1.0 / thb_width
thb_height = int(top_height/rate)

small_image = image.resize((thb_width, thb_height))
file_name = f'{slide_id}_[{ref}].png'
file_name = file_name.replace('/', ',')
small_image.save(file_name)
#color = ['blue', 'red']
draw = ImageDraw.Draw(small_image)

width = int(thb_width*1.0 / top_width * size[0])
height = width

cmaps = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'gold', 'KHAKI']

for cl in cluster:
    for img_path in cluster[cl]:
        label = int(cl)
        basename = os.path.basename(img_path)
        root_name = os.path.splitext(basename)[0]
        array = root_name.split("_")
        x = int(array[1][4:])
        y = int(array[2][4:])
        x = int(x / rate)
        y = int(y / rate)
        draw.rectangle([(x, y),(x+width-1, y+height-1)], outline=cmaps[label])

file_name = f'{slide_id}_box_[{ref}].png'
file_name = file_name.replace('/', ',')
small_image.save(file_name)

hyp = hyp.split()

for i in range(len(att_lt) - 1):
    base_layer = image.resize((thb_width, thb_height))
    color_layer = Image.new('RGBA', base_layer.size, (0,0,0))
    draw = ImageDraw.Draw(color_layer, 'RGBA')
    
    att = att_lt[i]
    j = 0
    for cl in cluster:
        opaque = int(att[j] * 255)
        for img_path in cluster[cl]:
            label = int(cl)
            basename = os.path.basename(img_path)
            root_name = os.path.splitext(basename)[0]
            array = root_name.split("_")
            x = int(array[1][4:])
            y = int(array[2][4:])
            x = int(x / rate)
            y = int(y / rate)
            draw.rectangle([(x, y),(x+width-1, y+height-1)], outline=cmaps[label])
            draw.rectangle([(x, y),(x+width-1, y+height-1)], fill=(opaque,opaque,opaque))
            
        j += 1
    small_image = Image.blend(base_layer, color_layer, alpha=0.7)
    file_name = f'{slide_id}_hyp_[{hyp[i]}].png'
    file_name = file_name.replace('/', ',')
    small_image.save(file_name)



