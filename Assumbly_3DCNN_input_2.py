# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 02:38:04 2019

@author: Jian Zhou
"""

import torch
import torch.optim as optim
from Load_inputs import Dataloder # Load the whole video
from model import Net
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import Counter
#from Model3D import NET
from Model3DTest import NET
import torchvision
from torchvision import transforms

from PIL import Image
import os
import os.path

dataloader = Dataloder()

inputloader, Num_input = dataloader.getloader()
# Num_input: Total number of input images
print('Num of input images = ', Num_input, '\n')

########## Main function ##############
t1 = time.time()

Y=96 # For E040-44 96*104; For E68-72: 74*94; For E63-67: 60*78. For E047-49: 104*184; For E075-78:90*66; For E059-61:118*102;
X=104

InputImags = np.zeros(((Num_input, Y, X)))

for i, data in enumerate(inputloader, 0):# of total frames, batch size is 1
    # i iteration starts from 0 !
    inputs = data
    #InputImags[i,:,:] = inputs[0,0,:,1:79] # For 60*79. # 1:79 Actually 1:78 are adopted!
    #InputImags[i,:,:] = inputs[0,0,7:73,5:95] # For 70*79
    InputImags[i,:,:] = inputs[0,0,:,:]
    #InputImags[i,:,:] = inputs[0,0,1:103,1:119] # For 70*79
    #InputImags[i,:,:] = inputs[0,0,5:79,9:103] # For 84*112
    #InputImags[i,:,:] = inputs[0,0,:,1:95] # For 74*95
    

NewInpVid=InputImags
t2 =time.time()
print('Loading time is {:.4f} sec'.format(t2-t1),'\n')

Num_Sepa=30 # 30 selected frames in a temporal order are assembled as a dataset for training 3D CNN model
Step=7
Num_Ite=35 # # of datasets are assembled in each video

t3 = time.time()

INPUT = np.zeros(((1, Y, X)))
for i in range (Num_Ite):
    for j in range (Num_Sepa):
        TMPIMG = NewInpVid[Step*j+i,:,:][np.newaxis]
        INPUT = np.append(INPUT, TMPIMG, axis=0)
        # INPUT = torch.cat((INPUT, TMPIMG), 0)
Input = INPUT[1:INPUT.shape[0],:,:] # All training datasets are stored in INPUTS continuously
#print('  Size of Input is ', Input.shape, ', Type of Input is ', type(Input))

np.save('Input_Max15_30_7_35_E44_ExaNor.npy', Input)

t4 = time.time()
print('Restaging time is {:.4f} sec'.format(t4-t3),'\n')