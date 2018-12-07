# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 13:29:09 2018

@author: Owen
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import glob
import pdb

class TrainDataset(Dataset):
    
    def __init__(self, img_dir, mask_dir, sf_dir, transform = None):
        img_path_list = []
        for img_path in glob.glob(img_dir):
            img_path_list.append(img_path)
        
        mask_path_list = []
        for mask_path in glob.glob(mask_dir):
            mask_path_list.append(mask_path)
        
        sf_path_list = []
        for sf_path in glob.glob(sf_dir):
            sf_path_list.append(sf_path)
        
        self.img_path_list = img_path_list
        self.mask_path_list = mask_path_list
        self.sf_path_list = sf_path_list
        self.len = len(self.img_path_list)
        self.transform = transform
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img, sf, mask = np.asarray(Image.open(self.img_path_list[idx])), np.asarray(Image.open(self.sf_path_list[idx])), np.asarray(Image.open(self.mask_path_list[idx]))
        img_tensor, sf_tensor, mask_tensor = torch.from_numpy(img).transpose(0,2).transpose(1,2).float(), torch.from_numpy(sf).transpose(0,2).transpose(1,2).float(), torch.from_numpy(mask).unsqueeze(0).float()
        return img_tensor, sf_tensor, mask_tensor


'''
if __name__ == '__main__':
    img_dir, mask_dir, sf_dir = 'data/inpainted_images/*.png', 'data/gauss_masks/*.png', 'data/surface_normals/*.png'
    dataset = TrainDataset(img_dir, mask_dir, sf_dir)
    loader = DataLoader(dataset, batch_size=32)
    for idx, data in enumerate(loader):
        img, sf, mask = data
        print(idx, img.shape, sf.shape, mask.shape)
'''


    
    

    
    
