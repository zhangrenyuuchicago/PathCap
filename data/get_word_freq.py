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

sorted_word = sorted(word_freq.items(), key=lambda kv: kv[1])

fout = open('sorted_word_freq.txt', 'w')
for item in sorted_word:
    fout.write(str(item) + '\n')

fout.close()



