# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:10:19 2019

@author: Jian Zhou
"""

class UNet_down_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super(UNet_down_block, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, 5, padding=2, bias = False)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 5, padding=2, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, 5, padding=2, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)
        self.reduce = torch.nn.Conv2d(input_channel, input_channel, 5, stride=2, padding=2, bias=False)
        self.down_size = down_size

    def forward(self, x):
        if self.down_size:
            x = self.reduce(x)
        x = torch.nn.functional.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = torch.nn.functional.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = torch.nn.functional.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        return x

class UNet_up_block(torch.nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel):
        super(UNet_up_block, self).__init__()
        self.up_sampling = torch.nn.ConvTranspose2d(input_channel, input_channel,4,2,1)
        self.conv1 = torch.nn.Conv2d(prev_channel + input_channel, output_channel, 5, padding=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 5, padding=2, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, 5, padding=2, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)

    def forward(self, prev_feature_map, x):
        x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = torch.nn.functional.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = torch.nn.functional.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = torch.nn.functional.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        return x


class UNet(torch.nn.Module):
    def __init__(self, opts):
        super(UNet, self).__init__()

        self.opts = opts
        
        # Encoder network
        self.down_block1 = UNet_down_block(1, 64, False)
        self.down_block2 = UNet_down_block(64, 128, True)
        self.down_block3 = UNet_down_block(128, 256, True)

        # bottom convolution
        self.mid_conv1 = torch.nn.Conv2d(256, 256, 5, padding=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(256)
        self.mid_conv2 = torch.nn.Conv2d(256, 256, 5, padding=4, dilation=2, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(256)
        self.mid_conv3 = torch.nn.Conv2d(256, 256, 5, padding=8, dilation=4, bias=False) # Keep dimension 
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.mid_conv4 = torch.nn.Conv2d(256, 256, 5, padding=8, dilation=4, bias=False)
        self.bn4 = torch.nn.BatchNorm2d(256)
        self.mid_conv5 = torch.nn.Conv2d(256, 256, 5, padding=2, bias=False) # Keep dimension 
        self.bn5 = torch.nn.BatchNorm2d(256)

        # Decoder network
        self.up_block2 = UNet_up_block(128, 256, 128)
        self.up_block3 = UNet_up_block(64, 128, 64)

        # Final output
        self.last_conv1 = torch.nn.Conv2d(64, 64, 5, padding=2, bias=False)
        self.last_bn = torch.nn.BatchNorm2d(64)
        self.last_conv2 = torch.nn.Conv2d(64,1,5,padding=2)

    def forward(self, x, test=False):
               
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)

        self.x4 = torch.nn.functional.leaky_relu(self.bn1(self.mid_conv1(self.x3)), 0.2)
        self.x4 = torch.nn.functional.leaky_relu(self.bn2(self.mid_conv2(self.x4)), 0.2)
        self.x4 = torch.nn.functional.leaky_relu(self.bn3(self.mid_conv3(self.x4)), 0.2)
        self.x4 = torch.nn.functional.leaky_relu(self.bn4(self.mid_conv4(self.x4)), 0.2)
        self.x4 = torch.nn.functional.leaky_relu(self.bn5(self.mid_conv5(self.x4)), 0.2)
 
        out = self.up_block2(self.x2, self.x4)
        out = self.up_block3(self.x1, out)

        out = torch.nn.functional.relu(self.last_bn(self.last_conv1(out)))
        out = torch.nn.functional.sigmoid(self.last_conv2(out)) #Try tanh and scale
            
        return out