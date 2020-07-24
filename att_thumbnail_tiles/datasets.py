import torch
from torch.utils.data import Dataset
import h5py
import json
import os
from PIL import Image
import glob
import random

class CaptionDataset(Dataset):
    def __init__(self, data_folder, split, transform, tile_transform, tile_folder, cluster_folder):
        self.split = split
        assert self.split in {'train', 'val', 'test'}
        self.data_folder = data_folder
        
        with open(os.path.join(data_folder, self.split + '_enc_captions.json'), 'r') as j:
            self.captions = json.load(j)
        with open(os.path.join(data_folder, self.split + '_enc_caplens.json'), 'r') as j:
            self.caplens = json.load(j)
        with open(os.path.join(data_folder, self.split + '_enc_imgs_lt.json'), 'r') as j:
            self.imgs_lt = json.load(j)
        
        self.slide_id = []
        for i in range(len(self.imgs_lt)):
            basename = self.imgs_lt[i]
            name_root = os.path.splitext(basename)[0]
            self.slide_id.append(name_root)
            full_path = os.path.join(self.data_folder, 'resized' ,self.imgs_lt[i])
            self.imgs_lt[i] = full_path
        
        self.slide_cluster = []
        for i in range(len(self.slide_id)):
            slide_id = self.slide_id[i]
            cluster_name = os.path.join(cluster_folder, slide_id + '_slide_tile_cluster.json')
            if not os.path.exists(cluster_name):
                self.slide_cluster.append(None)
            else:    
                with open(cluster_name, 'r') as fin:
                    cluster = json.load(fin)
                    self.slide_cluster.append(cluster)
        
        self.slide_site = {}
        for tile_path in glob.glob(tile_folder + '/*/*.jpg'):
            basename = os.path.basename(tile_path)
            array = basename.split('_')
            slide_id = array[0]
            if slide_id not in self.slide_site:
                array = os.path.split(os.path.dirname(tile_path))
                site = array[-1]
                self.slide_site[slide_id] = site

        for i in range(len(self.slide_cluster)):
            slide_id = self.slide_id[i]
            if not self.slide_cluster[i]:
                continue
            for cl in self.slide_cluster[i]:
                for j in range(len(self.slide_cluster[i][cl])):
                    tile_name = self.slide_cluster[i][cl][j]
                    tile_path = os.path.join(tile_folder, self.slide_site[slide_id], tile_name)
                    self.slide_cluster[i][cl][j] = tile_path
                    assert os.path.exists(tile_path)

        self.transform = transform
        self.tile_transform = tile_transform
        
        imgs_lt_tmp, captions_tmp, caplens_tmp, slide_cluster_tmp, slide_id_tmp = [], [], [], [], []
        for i in range(len(self.imgs_lt)):
            slide_id = self.slide_id[i]
            if slide_id in self.slide_site and self.slide_cluster[i]:
                imgs_lt_tmp.append(self.imgs_lt[i])
                captions_tmp.append(self.captions[i])
                caplens_tmp.append(self.caplens[i])
                slide_cluster_tmp.append(self.slide_cluster[i])
                slide_id_tmp.append(slide_id)

        self.imgs_lt = imgs_lt_tmp
        self.captions = captions_tmp
        self.caplens = caplens_tmp
        self.slide_cluster = slide_cluster_tmp
        self.slide_id = slide_id_tmp

        self.slide_proportion = []
        for i in range(len(self.slide_cluster)):
            proportion = [len(self.slide_cluster[i][j]) for j in self.slide_cluster[i]]
            sum_pro = sum(proportion)
            proportion = [proportion[j] / sum_pro for j in range(len(proportion))]
            self.slide_proportion.append(proportion)

        # print out slide number 
        print(f'slide tiles cluster: {len(self.slide_cluster)}')
        print(f'thumbnail number: {len(self.imgs_lt)}')
        
        self.dataset_size = len(self.captions)

    def __getitem__(self, index):
        img = Image.open(self.imgs_lt[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        slide_id = self.slide_id[index]
        caption = torch.LongTensor(self.captions[index])
        caplen = torch.LongTensor([self.caplens[index]])

        tiles = []
        if self.split == 'test' or self.split == 'val':
            for cl in self.slide_cluster[index]:
                img_name = random.choice(self.slide_cluster[index][cl])
                image = Image.open(img_name)
                image = self.tile_transform(image)
                tiles.append(image)
        else:
            cl_index = list(self.slide_cluster[index])
            random.shuffle(cl_index)
            for cl in cl_index:
                img_name = random.choice(self.slide_cluster[index][cl])
                image = Image.open(img_name)
                image = self.tile_transform(image)
                tiles.append(image)
        
        tiles = torch.stack(tiles)
        proportion = self.slide_proportion[index]
        proportion = torch.FloatTensor(proportion)

        if self.split is 'train':
            return img, caption, caplen, tiles, proportion
        elif self.split is 'val':
            all_captions = torch.LongTensor(
                self.captions[index:index+1])
            return img, caption, caplen, all_captions, tiles, proportion
        else:
            all_captions = torch.LongTensor(
                self.captions[index:index+1])
            return img, caption, caplen, all_captions, tiles, slide_id, proportion

    def __len__(self):
        return self.dataset_size
