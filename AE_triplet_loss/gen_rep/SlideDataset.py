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
from scipy.misc import imread
import json


class SlideDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, dir, transform):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            transform: image transformer.
        """
        self.ids = glob.glob(dir + "/*/*.jpg")
        print( "ids image number: " + str(len(self.ids)))
                
        ids_tmp = []

        mask_id = []
        with open('../train_enc_imgs_lt.json', 'r') as fin:
            mask_id = json.load(fin)
        with open('../val_enc_imgs_lt.json', 'r') as fin:
            mask_id += json.load(fin)
        with open('../test_enc_imgs_lt.json', 'r') as fin:
            mask_id += json.load(fin)
        mask_id_tmp = [slide_id[:-4] for slide_id in mask_id]
        mask_id = set(mask_id_tmp)
        print(f'mask id: {len(mask_id)}')

        for path in self.ids:
            basename = os.path.basename(path)
            array = basename.split('_')
            slide_id = array[0]
            if slide_id in mask_id:
                ids_tmp.append(path)

        self.ids = ids_tmp
        
        print( "ids image number(after filter): " + str(len(self.ids)))
        
        self.transform = transform
        print( "Initialize end")

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        img_name = self.ids[index]
        image = Image.open(img_name)
        if self.transform is not None:
            image = self.transform(image)
        basename = ntpath.basename(img_name)
        return image, basename 
                
    def __len__(self):
        return len(self.ids)


