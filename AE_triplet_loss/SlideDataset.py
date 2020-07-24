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


class SlideDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, dir, transform):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            transform: image transformer.
        """
        
        print('init from the beginning')
        self.__init_ids__(dir)
        print('build adj images')
        print(f'tile numbers after filtered: {len(self.ids)}')
        
        #self.slide_id = [sid for sid in self.slide_imgs]
        self.transform = transform
        print( "Initialize end")

    def __init_ids__(self, dir):
        
        self.ids = glob.glob(dir + "/*/*.jpg")
        print( "ids image number: " + str(len(self.ids)))
        
        '''
        ids_tmp = []
        for path in self.ids:
            basename = os.path.basename(path)
            array = basename.split('_')
            slide_id = array[0]
            ids_tmp.append(path)
        self.ids = ids_tmp
        '''

        self.slide_imgs = {}
        for path in self.ids:
            basename = os.path.basename(path)
            array = basename.split('_')
            slide_id = array[0]
            if slide_id in self.slide_imgs:
                self.slide_imgs[slide_id].append(path)
            else:
                self.slide_imgs[slide_id] = [path]
        
        print(f'tile numbers: {len(self.ids)}')
        print(f'slide numbers: {len(self.slide_imgs)}')

        self.adj_imgs = [[] for i in range(len(self.ids))]

        for i in tqdm.tqdm(range(len(self.ids))):
            path = self.ids[i]
            basename = os.path.basename(path)
            basename = basename[:-4]
            array = basename.split('_')
            slide_id = array[0]
            
            if os.path.exists('./adj_img/' + basename + '.txt'):
                fin = open('./adj_img/' +  basename + '.txt', 'r')
                while True:
                    line = fin.readline().strip()
                    if not line:
                        break
                    self.adj_imgs[i].append(line)
                fin.close()
        
        adj_imgs_tmp, ids_tmp = [], []

        for i in range(len(self.ids)):
            if len(self.adj_imgs[i]) > 0:
                ids_tmp.append(self.ids[i])
                adj_imgs_tmp.append(self.adj_imgs[i])

        self.ids = ids_tmp
        print(f'tile num: {len(self.ids)}')
        self.adj_imgs = adj_imgs_tmp
 

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        img_name = self.ids[index]
        image = Image.open(img_name)
        if self.transform is not None:
            image = self.transform(image)
        basename = ntpath.basename(img_name)
        
        pos_img_name = random.choice(self.adj_imgs[index])
        pos_image = Image.open(pos_img_name)
        if self.transform is not None:
            pos_image = self.transform(pos_image)
        pos_basename = ntpath.basename(pos_img_name)
        
        #while True:
        rand_i = random.randint(0, len(self.ids)-1)
        neg_img_name = self.ids[rand_i]
        #    if neg_img_name in self.adj_imgs[index]:
        #        continue

        neg_image = Image.open(neg_img_name)
        if self.transform is not None:
            neg_image = self.transform(neg_image)
        neg_basename = ntpath.basename(neg_img_name)
        
        return image, pos_image, neg_image, basename, pos_basename, neg_basename
                
    def __len__(self):
        return len(self.ids)

