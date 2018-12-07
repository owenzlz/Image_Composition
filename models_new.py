# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 15:06:44 2018

@author: Owen
"""
import torch
import torch.nn.functional as F
from torch import nn
import pdb

''' img_sf -> z: 6x240x320 -> 128x1x1'''
class Encoder(nn.Module):
    	
	def __init__(self):
		super(Encoder, self).__init__()
		self.conv1 = nn.Conv2d(6, 16, 3, 2, 1)
		self.conv_bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, 3, 2, 1) # downsample
		self.conv_bn2  = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
		self.conv_bn3 = nn.BatchNorm2d(64)
		self.conv4 = nn.Conv2d(64,128, 3, 2, 1) # downsample
		self.conv_bn4 = nn.BatchNorm2d(128)
		self.conv5 = nn.Conv2d(128,128, 2, 4, 1) # downsample
		self.conv_bn5 = nn.BatchNorm2d(128)

		# mu and sigma
		self.conv6a = nn.Conv2d(128, 128, (4, 6), 1, 0) # downsample
		self.conv6b = nn.Conv2d(128, 128, (4, 6), 1, 0) # downsample
    
	def encode(self, img_sf_mask_encoding):
		# print('0: ', img_sf_mask_encoding.shape)
		feat_1 = F.relu(self.conv_bn1(self.conv1(img_sf_mask_encoding)))
		# print('1: ', feat_1.shape)
		feat_2 = F.relu(self.conv_bn2(self.conv2(feat_1)))   
		# print('2: ', feat_2.shape)
		feat_3 = F.relu(self.conv_bn3(self.conv3(feat_2)))     
		# print('3: ', feat_3.shape)
		feat_4 = F.relu(self.conv_bn4(self.conv4(feat_3)))   
		# print('4: ', feat_4.shape)
		feat_5 = F.relu(self.conv_bn5(self.conv5(feat_4)))   
		# print('5: ', feat_5.shape)
		mu_z = self.conv6a(feat_5)
		sig_z = self.conv6b(feat_5)
		# print('mu and sigma: ', mu_z.shape, sig_z.shape)
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

''' z -> pred_loc_heatmap: 128x1x1 -> 1x240x320'''
class Decoder(nn.Module):
    	
	def __init__(self):
		super(Decoder, self).__init__()
		self.trans_conv1 = nn.ConvTranspose2d(128, 128, (4, 6), 1, 0)
		self.trans_conv_bn1 = nn.BatchNorm2d(128)
		self.trans_conv2 = nn.ConvTranspose2d(128, 128, 2, 4, 1)
		self.trans_conv_bn2 = nn.BatchNorm2d(128)
		self.trans_conv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
		self.trans_conv_bn3 = nn.BatchNorm2d(64)
		self.trans_conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
		self.trans_conv_bn4 = nn.BatchNorm2d(32)
		self.trans_conv5 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
		self.trans_conv_bn5 = nn.BatchNorm2d(16)		
		self.trans_conv6 = nn.ConvTranspose2d(16, 16, 4, 2, 1)
		self.trans_conv_bn6 = nn.BatchNorm2d(16)	
		self.trans_conv7 = nn.ConvTranspose2d(16, 1, 3, 1, 1)

	def forward(self, z): 
		# print('0: ', z.shape)
		feat_1 = F.relu(self.trans_conv_bn1(self.trans_conv1(z)))
		# print('1: ', feat_1.shape)
		feat_2 = F.relu(self.trans_conv_bn2(self.trans_conv2(feat_1)))
		feat_2 = F.interpolate(feat_2, size=(15,20)) 
		# print('2: ', feat_2.shape)
		feat_3 = F.relu(self.trans_conv_bn3(self.trans_conv3(feat_2))) 
		# print('3: ', feat_3.shape)
		feat_4 = F.relu(self.trans_conv_bn4(self.trans_conv4(feat_3)))
		# print('4: ', feat_4.shape) 
		feat_5 = F.relu(self.trans_conv_bn5(self.trans_conv5(feat_4)))
		# print('5: ', feat_5.shape) 		
		feat_6 = F.relu(self.trans_conv_bn6(self.trans_conv6(feat_5)))
		# print('6: ', feat_6.shape) 
		pred_mask = self.trans_conv7(feat_6)
		# print('7: ', pred_mask.shape)
		return pred_mask

'''
if __name__ == '__main__':
	# image encoder
	encoder = Encoder().cuda()
	img = torch.ones((1,3,240,320)).cuda()
	sf = torch.ones((1,3,240,320)).cuda()
	img_sf = torch.cat([img, sf], 1)
	encoding = encoder(img_sf)

	decoder = Decoder().cuda()
	pred_mask = decoder(encoding)

	print('img_sf: ', img_sf.shape, 'encoding: ', encoding.shape, 'pred_mask: ', pred_mask.shape)

	pdb.set_trace()	
'''
