import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import upsample

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(1, 4, 3)
        # 1 input image channel, 4 output channels, 3x3 square convolution
        #self.bn1 = nn.BatchNorm2d(4)
        #self.relu1 = nn.ReLU(inplace=True)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(4, 16, 3)
        #self.bn2 = nn.BatchNorm2d(16)
        #self.relu2 = nn.ReLU(inplace=True)
        #self.conv3 = nn.Conv2d(16, 3, 3)
        self.conv1 = self.double_conv(1,32)
        self.pool1 = nn.MaxPool2d(2) # the size of window to take a max (pool) over
        self.conv2 = self.double_conv(32,64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = self.double_conv(64,128)
        self.upsample = nn.Upsample(scale_factor=2,  mode='bilinear')
        self.dconv_up2 = self.double_conv(64+128, 64)
        self.dconv_up1 = self.double_conv(64+32, 32)
        self.out = nn.Conv2d(32, 2, 1) 
        # projection to 2 probabilities for 2 labers: on the circle or outside the circle. 1 square convolution

    def forward(self, x):  # U-Net
        imsize = x.size()[2:]
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.relu1(x)
        #x = self.pool(x)
        #x = self.conv2(x)
        #x = self.bn2(x)
        #x = self.relu2(x)
        #x = self.conv3(x)
        #x = upsample(x, imsize,  mode='bilinear')
        conv1 = self.conv1(x) # x: [batch_size, In_channels, Height, Width]
        x = self.pool1(conv1)
        conv2 = self.conv2(x)
        x = self.pool2(conv2)
        x = self.conv3(x)
        x = self.upsample(x) 
        x = torch.cat([x, conv2], dim=1) # concatenate, reuse feature map in different steps
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        # concatenate, reuse feature map from the output of convolution layers in different steps
        x = self.dconv_up1(x)
        x = self.out(x)
        return x
    
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1), # Convolving kernel size is 3
                nn.BatchNorm2d(out_channels), # out_channels is number of features
                nn.ReLU(inplace=True), # Meaning of True??
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                )