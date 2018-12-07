# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 15:06:44 2018

@author: Owen
"""
import torch
import torch.nn.functional as F
from torch import nn

''' img -> img_encoding: 3x240x320 -> 256x15x20'''
class Img_Encode(nn.Module):
    
    def __init__(self):
        super(Img_Encode, self).__init__()
        self.conv1 = nn.Conv2d(6, 96, kernel_size=11, stride=4, padding=5)
        self.conv_bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.conv_bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv_bn3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv_bn4 = nn.BatchNorm2d(384)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        
    def forward(self, img_sf): 
        feat_1 = F.relu(self.conv_bn1(self.conv1(img_sf)))
        feat_1_down = F.max_pool2d(feat_1, 2)
        feat_2 = F.relu(self.conv_bn2(self.conv2(feat_1_down)))      
        feat_2_down = F.max_pool2d(feat_2, 2)
        feat_3 = F.relu(self.conv_bn3(self.conv3(feat_2_down)))        
        feat_4 = F.relu(self.conv_bn4(self.conv4(feat_3)))        
        img_encoding = self.conv5(feat_4)
        return img_encoding

''' mask -> mask_encoding: 1x240x320 -> 256x15x20'''
class Mask_Encode(nn.Module):
    
    def __init__(self):
        super(Mask_Encode, self).__init__()
        
    def forward(self, mask):
        mask_encoding = mask.view(mask.size(0),256,15,20)
        return mask_encoding

''' img_sf_mask_encoding -> z: 512x15x20 -> 128x1x1'''
class VAE_Encode(nn.Module):
    
    def __init__(self):
        super(VAE_Encode, self).__init__()
        self.conv1 = nn.Conv2d(256+256, 512, 3, 1, 1)
        self.conv_bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, 3, 2, 1) # downsample
        self.conv_bn2  = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 256, 3, 1, 1)
        self.conv_bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256,256,3, 2, 1) # downsample
        self.conv_bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, 2, 2, 1) # downsample
        self.conv_bn5 = nn.BatchNorm2d(256)        
        # mu and sigma
        self.conv6a = nn.Conv2d(256, 128, 3, 1, 0) # downsample
        self.conv6b = nn.Conv2d(256, 128, 3, 1, 0) # downsample
        
    def encode(self, img_sf_mask_encoding):
        feat_1 = F.relu(self.conv_bn1(self.conv1(img_sf_mask_encoding)))
        feat_2 = F.relu(self.conv_bn2(self.conv2(feat_1)))   
        feat_3 = F.relu(self.conv_bn3(self.conv3(feat_2)))     
        feat_4 = F.relu(self.conv_bn4(self.conv4(feat_3)))   
        feat_5 = F.relu(self.conv_bn5(self.conv5(feat_4)))
        mu_z = self.conv6a(feat_5)
        sig_z = self.conv6b(feat_5)
        return mu_z, sig_z
        
    def reparameterize(self, mu_z, sig_z):
        """"z = mean + eps * sigma where eps is sampled from N(0, 1)."""
        epsilon = torch.randn(mu_z.size(0), mu_z.size(1)).unsqueeze(2).unsqueeze(3).cuda()
        z = mu_z + epsilon*torch.exp(sig_z/2)
        return z  
    
    def forward(self, img_sf_mask_encoding):
        mu_z, sig_z = self.encode(img_sf_mask_encoding)
        z = self.reparameterize(mu_z, sig_z)   
        return z

''' z -> z_decoding: 128x1x1 -> 256x15x20'''
class Latent_Decode(nn.Module):
    
    def __init__(self):
        super(Latent_Decode, self).__init__()
        self.trans_conv1 = nn.ConvTranspose2d(128, 128, 3, 1, 0)
        self.trans_conv_bn1 = nn.BatchNorm2d(128)
        self.trans_conv2 = nn.ConvTranspose2d(128, 128, 2, 2, 1)
        self.trans_conv_bn2 = nn.BatchNorm2d(128)
        self.trans_conv3 = nn.ConvTranspose2d(128, 256, 3, 2, 1)
        self.trans_conv_bn3 = nn.BatchNorm2d(256)
        self.trans_conv4 = nn.ConvTranspose2d(256, 256, 3, 1, 1)
        self.trans_conv_bn4 = nn.BatchNorm2d(256)
        self.trans_conv5 = nn.ConvTranspose2d(256, 256, 3, 2, 1)
        self.trans_conv_bn5 = nn.BatchNorm2d(256)        
        self.trans_conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        
    def forward(self, z): 
        feat_1 = F.relu(self.trans_conv_bn1(self.trans_conv1(z)))
        feat_2 = F.relu(self.trans_conv_bn2(self.trans_conv2(feat_1)))     
        feat_3 = F.relu(self.trans_conv_bn3(self.trans_conv3(feat_2)))     
        feat_4 = F.relu(self.trans_conv_bn4(self.trans_conv4(feat_3))) 
        feat_5 = F.relu(self.trans_conv_bn5(self.trans_conv5(feat_4)))
        z_decoding = F.interpolate(self.trans_conv6(feat_5), size=(15,20))
        return z_decoding

''' z_img_sf_encoding -> pred_mask: 512x15x20 -> 1x240x320'''
class VAE_Decode(nn.Module):
    
    def __init__(self):
        super(VAE_Decode, self).__init__()
        self.trans_conv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.trans_conv_bn1 = nn.BatchNorm2d(256)
        self.trans_conv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.trans_conv_bn2 = nn.BatchNorm2d(128)
        self.trans_conv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.trans_conv_bn3 = nn.BatchNorm2d(64)
        self.trans_conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.trans_conv_bn4 = nn.BatchNorm2d(32)
        self.trans_conv5 = nn.ConvTranspose2d(32, 1, 3, 1, 1)
        
    def forward(self, z): 
        feat_1 = F.relu(self.trans_conv_bn1(self.trans_conv1(z)))
        feat_2 = F.relu(self.trans_conv_bn2(self.trans_conv2(feat_1)))   
        feat_3 = F.relu(self.trans_conv_bn3(self.trans_conv3(feat_2))) 
        feat_4 = F.relu(self.trans_conv_bn4(self.trans_conv4(feat_3))) 
        pred_mask = self.trans_conv5(feat_4)
        return pred_mask

'''
if __name__ == '__main__':
    # image encoder
    img_encoder = Img_Encode()
    img = torch.ones((1,3,240,320))
    sf = torch.ones((1,3,240,320))
    img_sf = torch.cat([img, sf], 1)
    img_sf_encoding = img_encoder(img_sf)
    print('img_sf_encoding: ', img_sf_encoding.size())
    
    # mask encoder
    mask_encoder = Mask_Encode()
    mask = torch.ones((1,1,240,320))
    mask_encoding = mask_encoder(mask)
    print('mask_encoding: ', mask_encoding.size())
    
    # concat feature
    img_sf_mask_encoding = torch.cat([img_sf_encoding, mask_encoding], 1)
    print('img_mask_encoding: ', img_sf_mask_encoding.size())
    
    # encode img_mask_encoding
    vae_encoder = VAE_Encode()
    z = vae_encoder(img_sf_mask_encoding)
    print('z: ', z.size())
    
    # decode z 
    latent_decoder = Latent_Decode()
    z_decoding = latent_decoder(z)
    print('z_decoding: ', z_decoding.size())
    
    # decode z_img_sf_encoding
    z_img_sf_encoding = torch.cat([img_sf_encoding, z_decoding], 1)
    print('z_img_sf_encoding: ', z_img_sf_encoding.size())
    vae_decoder = VAE_Decode()
    pred_mask = vae_decoder(z_img_sf_encoding)
    print('pred_mask: ', pred_mask.size())
'''








