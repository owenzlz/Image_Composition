# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 21:45:50 2018

@author: Owen
"""
import torch
from torch.autograd import Variable
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time
from PIL import Image
from data_utils import *
from models_new import *
from vis_tools import *

display = visualizer(port = 8091)

img_dir, mask_dir, sf_dir = 'data/inpainted_images/*.png', 'data/gauss_masks/*.png', 'data/surface_normals/*.png'
dataset = TrainDataset(img_dir, mask_dir, sf_dir)
loader = DataLoader(dataset, batch_size=64, shuffle = True)

encoder = Encoder().cuda()
decoder = Decoder().cuda()

optimizer = torch.optim.Adam([{'params': encoder.parameters()}, 
                              {'params': decoder.parameters()}], lr=0.001)

celoss = nn.CrossEntropyLoss()
mse = nn.MSELoss()

step = 0
total_epoch = 3000
KLD_arr_np = np.array([])
mask_loss_arr_np = np.array([])
for epoch in range(total_epoch):
    start_time = time.time()
    for i, data in enumerate(loader):
        img, sf, mask = data
        img, sf, mask = Variable(img.cuda()), Variable(sf.cuda()), Variable(mask.cuda())
        
        # forward pass the network
        img_sf = torch.cat([img, sf], 1)
        encoding = encoder(img_sf)
        mu_z, sig_z = encoder.encode(img_sf)
        pred_mask = decoder(encoding)
        
        # compute the loss
        mask_loss = mse(pred_mask, mask)
        KLD = 0.5*torch.sum(mu_z**2 + torch.exp(sig_z) - sig_z - 1) # 0.5* sum(mu^2 + log(sigma^2) - sigma^2 - 1)        
        alpha = 0.1
        total_loss = (1-alpha)*mask_loss + alpha*KLD
        
        # backpropagate to update the network
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()   
        
        step += 1

        err_dict = {'KLD': KLD.cpu().data.numpy(),
                    'Reconstruction Loss': mask_loss.cpu().data.numpy()}
        display.plot_error(err_dict)
        
        if step % 10 == 0: 
            img_show = img.cpu().data.numpy()[0].astype(np.uint8)
            sf_show = sf.cpu().data.numpy()[0].astype(np.uint8)
            mask_show = mask.cpu().data.numpy()[0].astype(np.uint8)
            pred_mask_show = pred_mask.cpu().data.numpy()[0].astype(np.uint8)
            
            display.plot_img_255(img_show, win=1, caption = 'inpainted image')
            display.plot_img_255(sf_show, win=2, caption = 'surface normal')
            display.plot_img_255(mask_show, win=3, caption = 'GT mask')
            display.plot_img_255(pred_mask_show, win=4, caption = 'pred mask')    
            
            # mask_show = np.flip(mask_show, axis=1)
            # display.plot_heatmap(mask_show[0], win=5, caption = 'GT mask')     
            # pred_mask_show = np.flip(pred_mask_show, axis=1)
            # display.plot_heatmap(pred_mask_show[0], win=6, caption = 'pred mask')       
    
    KLD_np = KLD.cpu().data.numpy()
    mask_loss_np = mask_loss.cpu().data.numpy()
    KLD_arr_np = np.append(KLD_arr_np, KLD_np)
    mask_loss_arr_np = np.append(mask_loss_arr_np, mask_loss_np)
    end_time = time.time()    
    print(epoch, 'KLD: ', KLD_np, 'Recon: ', mask_loss_np, end_time - start_time)
    
x_index = np.arange(1,total_epoch+1,1)
plt.figure()
plt.plot(x_index, mask_loss_arr_np, 'r')
plt.plot(x_index, KLD_arr_np, 'b')
plt.legend(['mask_loss', 'KL divergence'])    
plt.title('Loss over Epoch')
plt.savefig('Loss over Epoch.png')


torch.save(encoder.state_dict(), 'models/encoder_001.pt')
torch.save(decoder.state_dict(), 'models/decoder_001.pt')


