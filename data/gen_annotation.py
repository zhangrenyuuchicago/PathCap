import csv
import tqdm
import glob
import ntpath
import sys
from shutil import copy
import json

pid_ann = {}
pid_organ = {}
pid_row = {}
with open('GTEx_v7_Annotations_SampleAttributesDS.txt', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    for row in csv_reader:
        pid = row[0]
        ann = row[3]

        if len(pid) > 10:
            array = pid.split('-')
            seq = array[1]
            try:
                num = int(array[2])
            except:
                continue
            if seq in pid_ann:
                if num in pid_ann[seq]:
                    #print("already exists")
                    #print(pid)
                    #pid_ann[seq][num].append(ann)
                    pass
                else:
                    pid_ann[seq][num] = ann
            else:
                pid_ann[seq] = {}
                pid_ann[seq][num] = ann


img_ann = {}

target_dir = './images/resized/'
fout = open('./images/annotations.txt', 'w')

for file_path in glob.glob(target_dir + "/*.png"):
    basename = ntpath.basename(file_path)
    basename_root = basename[:-4]
    
    #print(basename)
    pid_array = basename_root.split('-')
    if len(pid_array) < 3:
        continue

    new_pid = pid_array[1]
    new_num = int(pid_array[2]) + 1

    if new_pid in pid_ann:
        if new_num in pid_ann[new_pid]:
            caption = pid_ann[new_pid][new_num]
            new_num_str = str(new_num)
            if len(new_num_str) < 4:
                new_num_str = '0'*(4-len(new_num_str)) + new_num_str
            pid = new_pid + '-' + new_num_str
            file_name = basename
            fout.write(basename + '\t' + caption + '\n')
            img_ann[basename] = caption
fout.close()

from collections import Counter
import nltk

word_freq = Counter()

for img_pid in img_ann:
    caption = img_ann[img_pid]
    tokens = nltk.tokenize.word_tokenize(str(caption).lower())
    word_freq.update(tokens)

min_word_freq = 5
words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
print(f'chosen words num: {len(words)}')

word_map = {k: v+1 for v, k in enumerate(words)}
word_map['<unk>'] = len(word_map) + 1
word_map['<start>'] = len(word_map) + 1
word_map['<end>'] = len(word_map) + 1
word_map['<pad>'] = 0

with open('images/word_map.json', 'w') as fout:
    json.dump(word_map, fout)

max_len = 53

enc_imgs = []
enc_captions = []
caplens = []

fout = open('images/annotations_enc.txt', 'w')

for img_pid in img_ann:
    ann = img_ann[img_pid]
    tokens = nltk.tokenize.word_tokenize(str(ann).lower())
    enc_ann = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in tokens] + [word_map['<end>']] + [word_map['<pad>']] * (max_len - len(tokens))
    #print(enc_ann)
    proc_ann = ['<start>'] 
    for word in tokens:
        if word in word_map:
            proc_ann.append(word)
        else:
            proc_ann.append('<unk>')
    proc_ann.append('<end>')
    fout.write(img_pid + '\t' + ' '.join(proc_ann) + '\n')

    c_len = len(tokens) + 2
    enc_imgs.append(img_pid)
    enc_captions.append(enc_ann)
    caplens.append(c_len)

fout.close()

print(f'enc imgs: {len(enc_imgs)}')
print(f'enc captions: {len(enc_captions)}')

assert len(enc_imgs) == len(enc_captions)

train_enc_imgs = []
train_enc_captions = []
train_caplens = []

val_enc_imgs = []
val_enc_captions = []
val_caplens = []

test_enc_imgs = []
test_enc_captions = []
test_caplens = []

import random

for i in range(len(enc_imgs)):
    ch = random.random()

    if ch < 0.8:
        train_enc_imgs.append(enc_imgs[i])
        train_enc_captions.append(enc_captions[i])
        train_caplens.append(caplens[i])
    elif ch < 0.9:
        val_enc_imgs.append(enc_imgs[i])
        val_enc_captions.append(enc_captions[i])
        val_caplens.append(caplens[i])
    else:
        test_enc_imgs.append(enc_imgs[i])
        test_enc_captions.append(enc_captions[i])
        test_caplens.append(caplens[i])

print(f'max len: {max_len}')

with open('images/train_enc_imgs_lt.json', 'w') as fout:
    json.dump(train_enc_imgs, fout)

with open('images/train_enc_captions.json', 'w') as fout:
    json.dump(train_enc_captions, fout)

with open('images/train_enc_caplens.json', 'w') as fout:
    json.dump(train_caplens, fout)

with open('images/test_enc_imgs_lt.json', 'w') as fout:
    json.dump(test_enc_imgs, fout)

with open('images/test_enc_captions.json', 'w') as fout:
    json.dump(test_enc_captions, fout)

with open('images/test_enc_caplens.json', 'w') as fout:
    json.dump(test_caplens, fout)

with open('images/val_enc_imgs_lt.json', 'w') as fout:
    json.dump(val_enc_imgs, fout)

with open('images/val_enc_captions.json', 'w') as fout:
    json.dump(val_enc_captions, fout)

with open('images/val_enc_caplens.json', 'w') as fout:
    json.dump(val_caplens, fout)


