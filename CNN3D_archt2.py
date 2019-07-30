# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:22:42 2019

@author: Jian Zhou
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import upsample

class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.conv_layer1 = self.double_conv(1,32)
        #self.pool1 = nn.MaxPool3d((2,2,2), stride=(2,2,2))
        self.reduce1 = nn.Conv3d(32,32,3,2,1)       
        self.conv_layer2 = self.double_conv(32,64)       
        #self.reduce2 = nn.Conv3d(64,64,3,2,1)
        #self.conv_layer3 = self.double_conv(64,128) 
        self.upsample = nn.Upsample(scale_factor=2,  mode='trilinear')
        
        self.dconv_up1 = self.double_conv(64, 32)
        #self.dconv_up1 = self.double_convRed(64, 32)
        #self.reduce2 = nn.Conv3d(32,32,(3,3,3),stride=(2,1,1), padding=(1,1,1))
        #self.dconv_up2 = self.double_convRed(32, 32)
        self.out = nn.Conv3d(32, 2, (30,1,1)) 
        #self.last_conv = nn.Conv3d(32, 2, (15,3,3), padding=(0,1,1))

    def forward(self, x):  # Encoder network and Decoder newtowrk
        imsize = x.size()[2:]  
        
        conv1 = self.conv_layer1(x) # x: [batch_size, In_channels, Height, Width]
        #x = self.pool1(conv1)
        x= self.reduce1(conv1) # Encoder network: downsampling by Conv3d
        x = self.conv_layer2(x)
        #x = self.reduce2(x)
        #x = self.conv_layer3(x)
        x = self.upsample(x) # Decoder network: Upsampling  
        x = self.dconv_up1(x)
        #x = self.dconv_up2(x)
        #x= self.reduce2(x)        
        x = self.out(x)       
        #out = self.last_conv(x) #Add based on Liang 
        out = F.sigmoid(x)
        
        return out
    
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, (1,3,3), padding=(0,1,1)), 
                nn.BatchNorm3d(out_channels), 
                nn.ReLU(inplace=True), 
                nn.Conv3d(out_channels, out_channels, (1,3,3), padding=(0,1,1)),
                nn.BatchNorm3d(out_channels),
                # nn.ReLU(inplace=True), # Adding based on Liang
                )