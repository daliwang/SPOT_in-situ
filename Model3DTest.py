# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:58:05 2019

@author: Jian Zhou
"""
import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
from torch.nn.functional import upsample

class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        #self.conv1 = nn.Conv2d(1, 4, 3)
        # 1 input image channel, 4 output channels, 3x3 square convolution
        self.conv_layer1 = self.double_conv(1,32)
        self.pool1 = nn.MaxPool3d((2,2,2), stride=(2,2,2)) # Check whether it is subsampled by 2 ?
        # Set for D_in = 388        
        #self.conv_layer2 = self.double_conv(32,64)
        #self.pool2 = nn.MaxPool3d((2,2,2), stride=(2,2,2))
        # Set for D_in = 388
        self.conv_layer2 = self.double_conv(32,64)
        self.upsample = nn.Upsample(scale_factor=2,  mode='trilinear')
        # nearest neighbor and linear, bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor, respectively.
        # Check whether it is upsampled by 2
        self.dconv_up2 = self.double_conv(64+32, 32)
        #self.dconv_up1 = self.double_conv(64+32, 32)
        self.out = nn.Conv3d(32, 2, (30,1,1)) 
        # projection to 2 probabilities for 2 labers: on the circle or outside circle. 1 square convolution for H and W
        # 80 for Depth dimension. Outsizes:[1,2,1,96,104]

#############        
        #self.keep=self.conv_layer1.weight.data # which I added        
        #self.keep=self.conv_layer1.parameters()
        #w = list(self.conv_layer1.parameters())
        #print('conv_laryer1 weights are', w)
        #W=np.array(w)
        #print('Shape of W is ', W.shape,' Type of W is ', type(W))
        #print('Type of W[1] is ', type(W[1]))# <class 'torch.nn.parameter.Parameter'>
        #print('Max of Conv1 weights is', max(w), 'Min of Conv1 weights is ', min(w))

        #w_1=list(self.out.parameters())
        #W_1=np.array(w_1)
        #print('Shape of W_1 is ', W_1.shape,' Type of W is ', type(W_1))
        #print(' Max of W_1 is ', np.amax(W_1[1,:]))
        #print('Max of out weights is', max(w_1),'Min of Conv1 weights is ', min(w_1))
##############
        
    def forward(self, x):  # U-Net
        imsize = x.size()[2:]
        # Check what's the meaning of this ??     
        # 3DCNN input: [N, C_in, Dep_in, Hei_in, Wid_in]
        # output: [N, C_out, Dep_out, Hei_out, Wid_out]   
        conv1 = self.conv_layer1(x) # x: [batch_size, In_channels, Height, Width]
        #x= self.reduce()
        x = self.pool1(conv1)
        #conv2 = self.conv_layer2(x)
        #x = self.pool2(conv2)
        x = self.conv_layer2(x)
        
        x = self.upsample(x) 
        x = torch.cat([x, conv1], dim=1) # concatenate, reuse feature map in different steps
        x = self.dconv_up2(x)
        #x = self.upsample(x)
        #x = torch.cat([x, conv1], dim=1)
        # concatenate, reuse feature map from the output of convolution layers in different steps
        #x = self.dconv_up1(x)
        x = self.out(x)
        
        # which I added to test whether the weight has changed
        #if (self.conv_layer1.weight.data==self.keep).all(): print('same!')
        #w_1 = list(self.conv_layer1.parameters())
        
        #w_1=list(self.out.parameters())
        #print('out weights are', w_1)
        #if (self.conv_layer1.parameters()==self.keep).all(): print('same!')
        #else: print('Weights are', w_1)     
        
        return x
    
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, (1,3,3), padding=(0,1,1)), # Filter size is 1 means no filter. 
                # Don't conversion on depth dimention
                nn.BatchNorm3d(out_channels), # out_channels is number of features
                nn.ReLU(inplace=True), # Meaning of True??
                nn.Conv3d(out_channels, out_channels, (1,3,3), padding=(0,1,1)),
                nn.BatchNorm3d(out_channels),
                )