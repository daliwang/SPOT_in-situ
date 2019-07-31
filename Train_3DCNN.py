# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:34:15 2019

@author: Jian Zhou
"""

import torch
import torch.optim as optim
from Load_inputs import Dataloder # Load the whole video
#from model import Net
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import Counter
#from Model3D import NET
#from CNN3D_archt2 import NET
from Model3DTest import NET
import torchvision
from torchvision import transforms

from PIL import Image
import os
import os.path
import random

# net = Net()
# print(net)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

Net3D = NET()
# Net3D = Net3D.double()
'''
dataloader = Dataloder()

inputloader, Num_input = dataloader.getloader()
# Num_input: Total number of input images
print('Num of input images = ', Num_input, '\n')
'''

criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.3,1.0]))# Initial: 0.005,0.1. Real(0) 8293, (1)1691
# 80 out of 401 frames from E044_Max_13 over all time period 
  # 0.15, 1.0, Epo 35 LrINI=0.01 lr=0.01, Lr_decay=10: 0.993, 0.965; 8185(8197), 1679(1787);
  
  # 0.3, 1.0, Epo 35 LrINI=0.01 lr=0.01, Lr_decay=10: 0.995, 0.975; 8219(8230), 1680(1754); 
            # Epo 35 lrINI=0.1  lr=0.01, Lr_dacay=10: 0.981, 0.913; 8034(8086), 1639(1898); 
            # Epo 60 lrINI=0.01 lr=0.01, Lr_dacay=15: 0.996, 0.980; 8235(8244), 1682(1740). BEST!
            
            # Epo 35 LrINI=0.01 lr=0.01, Lr_decay=5: 0.984, 0.920; 8159(8296), 1554(1688);# lr becomes useless when it's too smaller. lr should NOT be used for too short epoch
            # Epo 60(35 is enough!) LrINI=0.01, lr=0.01, Lr_decay=15: 0.996, 0.978; 8225(8233), 1683(1751);
  # 0.4, 1.0, Epo 35 LrINI=0.01 lr=0.01, Lr_decay=10: 0.994, 0.971; 8220(8247), 1664(1737);

# First 80 out of 402 frames from E040_Min_13
  # 0.3, 1.0, Epo 35 lrINI=0.01 lr=0.01, Lr_dacay=15: 0.999, 0.973; 9476(9497), 481(487).
    
          
optimizer = optim.SGD(Net3D.parameters(), lr=0.01, momentum=0.9)
#Stochastic Gradient Descent simplest update weights   

def adjust_lr(optimizer, epoch, lr_decay=15):
    lr = 0.01 *(0.1**((epoch)//lr_decay))# epoch-1
    if (epoch) % lr_decay == 0: # epoch-1
        print('LR is set to: ', lr)
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_acc(pred, label): # calculate accuracy
    pred = pred.unsqueeze(1)
    # Increase a dimension of pred in order to compare with label
    #label = label.unsqueeze(1)# Extend 1 dimension because it has one less dimension compared with pred!
    
    # print('  pred size in comput_acc:', pred.shape)
    # print('  label size in comput_acc:', label.shape,'\n')
    # print(' '.join('%5s' % classes[pred[j]] for j in range(2)))
    
    acc_sum = (pred == label).sum()
    # Totally 96*104*batchsize. predictions/comparisons.
    return acc_sum.item()


def compute_F(pred, label): # calculate F1 scores for 2 labels
    pred = pred.unsqueeze(1).numpy()
    label = label.numpy()
    # Make pred and label have same dimensions, 5 dimensions
    
    # print('pred size in compute_F = ', pred.shape)
    # print('label size in compute_F = ', label.shape,'\n')
    
    # print(Counter(label.reshape(-1)))
    # Totally 96*104*batch_size predictions/comparisons
    Pred_Tru = []
    # All returned true
    Real_Tru = []
    # All real true
    Corr_Tru = []
    # Correct predicted true
         
    for i in range(2):# Based on the number of classes
        Pred_Tru.append((pred == i).sum())
        Real_Tru.append((label == i).sum())
        Corr_Tru.append(((pred == i)*(label == i)).sum())
        # classes = ['outside','onside'] deined in find_classes():
    
    return Pred_Tru, Real_Tru, Corr_Tru

def Read_label(labeldir):
    LABELS = []
    Labels = [] # np.zeros(((1,96,104)))
    CountLabl=0
    
    for filename in os.listdir(labeldir):
        LABELS.append(filename)
        CountLabl +=1
    for filename in os.listdir(labeldir):
        LABELS[int(filename[7:-4])-1]=filename# LABELS[0] stores 'GrdTruh13.png'. Here 7:-4 means 13.
        
    for i in range(CountLabl):
        _Label = os.path.join(labeldir, LABELS[i])# print(_image) # : Data\Inputs\Label\GrdTruhX.png. Labeldir is 'Data\\Inputs\\Label'
        _Labels = Image.open(_Label)     
        # print('   Type of _Labels is ', type(_Labels)) # Type is <class 'PIL.PngImagePlugin.PngImageFile'>
        # print('   Size of _Labels is ', _Labels.shape)
        
        _Labels = transforms.ToTensor()(_Labels)
        # print('Shape of _Labels is ', _Labels.shape)

        Labels.append(_Labels) # Labels' type is <class 'list'>       
    
    return Labels
        
################################################   
'''
    LABELS = []
    _label=os.path.join(labeldir,'GrdTruh.png')
    
    LABELS.append(_label)# class 'list'
    # print('shape of _label is ', np.shape(LABELS)) # (1,)

    _Labels = Image.open(LABELS[0])
    # Class list cannot use read 
    #LABELS=np.asarray(_Labels)
    #print('shape of LABELS is ', LABELS.shape)# [96,104]
    
    _Labels = transforms.ToTensor()(_Labels)
    # print('Shape of _Labels is ', _Labels.shape) [1,96,104]. Automatically add one dimension to _Labels
    # transforms.ToTensor() only works for numpy with 2 dimensons or 3 dimensons 
    return _Labels
'''
################################################


########## Main function ##############
# t3 = time.time()
'''
output_target = []  
net.load_state_dict(torch.load('model.ckpt'))# Load the trained 2DCNN model.
# This 2D CNN model is trained(290 frames) and tested(100 frames), for E044_Max_13.  

#INPTIMGS=np.zeros(((401, 96, 104)))
  
#for i, data in enumerate(inputloader, 0):
#    INPUTS = data
#    INPTIMGS[i,:,:] = INPUTS[0,0,:,:] # inputs_size:[1,1,96,104]
        
        #if i==3:
        #    plt.imshow(inputs[0,0,:,:])
        #    break
        ## Starts from i=183 to i=400, all images are zeros!!!!!?????
            
with torch.no_grad():
    net.eval()
    
    T_0 = 0
    Pred = []
    
# Based on trained 2DCNN model, predict on the whole video frames to identify central circle. Find T_0
    for i, data in enumerate(inputloader, 0):# of total frames, batch size is 1
        # i iteration starts from 0 !
        
        # print('Current image is ',i, '\n')
        inputs = data
        # inputs_size: [batch size, # of input channels(image value of each pixel), # of rows, # of columns]
        # [1,1,96,104]
        #plt.imshow(inputs[0,0,:,:])
        
        outputs = net(inputs)# Outputs_size: [1 batch size, 2 (Probility of being each class), 96 of rows, 104 of columns]
        _, pred = torch.max(outputs, dim=1)
        # pred_size: [1,96,104], Pick the highest probility of belonging to a class at each pixel
        
        # print('Orig output_size = ', outputs.size()) # [1,2,96,104]
        # print('Orig pred size = ', pred.size()) # [1,96,104]
        
        Pred = pred.numpy()# type is numpy.ndarray
        
        [Deph,Heit,Widt]=Pred.shape
        # pred_size is [1 (batch size),96,104]
        # plt.imshow(Pred[0,:,:])
        #if i==3:
        #    plt.imshow(inputs[0,0,:,:])
        #    break
        
        if Pred[0, int(Heit/2-1), int(Widt/2-1)] == 1:
            # Find the 1st predicted frame that has a circle in the center, i.e., if center posization (47,51) is 1.
            # 
            T_0 = i # Starts from 0!
            print ('T_0 = ',T_0, ', Count from 0', '\n')
            # print ('Size of selected Pred = ', Pred[0,:,:].shape)
            # plt.imshow(Pred[0,:,:])
            
            #print('Pred[0,47,51] = ', Pred[0,47,51], ', Max of Pred[0,:,:] = ', np.max(Pred[0,:,:]))
            # Check what's the range of Pred[]
            # Whether Prediction of a frame is in [0 ~ 1]!!!
            break
            
    PRED = np.zeros(((Num_input, Heit, Widt)))# type is numpy.ndarray
    # Store all predicted frames with pixels labeled as either 0 or 1
    # print('Shape of PRED = ', PRED.shape, end = '\n\n')
    # (401,96,104)
    
    NwMatrix = np.zeros(((Num_input - T_0, Heit, Widt)))
    # Store predicted frames starts from T_0
    
    NewMatrix = np.zeros(((1, Heit, Widt))) # Store predicted frames having central rectangle
    Fram = np.zeros((Heit, Widt))
    InputImags = np.zeros(((Num_input, Heit, Widt)))
    # Store all modified input images
    InptFram = np.zeros(((Num_input-T_0, Heit, Widt)))
    INPTFram = np.zeros(((1, Heit, Widt)))
    
    for i, data in enumerate(inputloader, 0):# i: Total # of input frames, because batch size is 1
        
        inputs = data
        
        InputImags[i,:,:] = inputs[0,0,:,:] # inputs_size:[1,1,96,104]. Range is in [0.0, 1.0]
        
        outputs = net(inputs)
        # print('Input type is ', type(inputs),', Output type is ', type(outputs))
        # input/output type is Totensor
        _, pred = torch.max(outputs, dim=1)
        
        PRED[i,:,:]=pred.numpy()
        # if PRED[i, int(Heit/2-1), int(Widt/2-1)] !=1:
        
        # Predict pixels on each frame for being 0 or 1    
            
        # print('Current iteration = ', i+1)
        
    # print('Type of inputs = ', type(inputs),'\n', 'Shape of inputs = ', inputs.shape) # [Tensor, [1,1,96,104]
    # print('\n','Type of outputs = ', type(outputs), '\n', 'Shape of outputs = ', outputs.shape) # Tensor, [1,2,96,104]
    print('Prediction based on trained 2DCNN is finished...', '\n')
    
    T_abnol = 0
    # Stores NO. of frame that is predicted abnormally, i.e., no center circle in the frame
    
######## Centering the rectangle of each frame after T_0
    for i in range (T_0, Num_input):# i from T_0 to (Num_input-1)  
        Flag = 0
        
        # print('Current iteration = ', i-T_0+1)
        if PRED[i, int(Heit/2-1), int(Widt/2-1)] != 1:
        # Find frames that don't have a central circle after T_0
            T_abnol = np.append(T_abnol,i) 
            # 1st item of T_abnol is 0, which should be disgarded. Before T_0 also should be recorded
            continue
        else:
            TemFra = InputImags[i,:,:][np.newaxis] # Make it to be 3 dimensions. for numpy.ndarray
            INPTFram = np.append(INPTFram, TemFra, axis=0)
    
    print('Shape of INPTFram = ', INPTFram.shape)
    
    IpVide = INPTFram[1:INPTFram.shape[0],:,:][np.newaxis][np.newaxis]
    print('Shape of IpVide = ', IpVide.shape)
        
print('    2DCNN predicted outliers are ', T_abnol)
# t4 = time.time()
print('Total time of 2D CNN based processing is {:.4f} sec'.format(t4-t3),'\n')

#NewInpVid=np.zeros(((Num_input-T_0,96,104)))
NewInpVid=np.zeros(((IpVide.shape[2],96,104)))

for i in range (IpVide.shape[2]):
    NewInpVid[i,:,:] = IpVide[0,0,i,:,:]
# for i in range (80):# Select every 4 frame to form the input for 3DCNN model
#     NewInpVid[i,:,:] = IpVide[0,0,4*i,:,:] # The last frame is No. 316, which starts from 0.
#NewInpVid[79,:,:]=IpVide[0,0,IpVide.shape[2]-1,:,:]

# for i in range(80):
#     NewInpVid[i,:,:] = IpVide[0,0,i,:,:]# First 80 video frames which have center nuggets as 3D inputs. Last 80 frames: -i-1
    # Range in [0.0,1.0]
# Labels = Read_label('Data\\Inputs\\Label')

NewInpVid = transforms.ToTensor()(NewInpVid[:,:,:])
# Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].

NewInpVid = NewInpVid.permute(1,2,0) # Original NewInpVid has 3 dimensions
print('Shape of NewInpVId = ', NewInpVid.shape) # [Num of frames,96,104]
          
#NewInpVid = NewInpVid.unsqueeze(0).unsqueeze(0)  #  for torch.tensor    
#print('Shape of NewInpVId = ', NewInpVid.shape, ', Type of NewInpVid = ', type(NewInpVid))# [1,1,80,96,104], Tensor

Num_Sepa=50 # Continuous 50 frames are used as a input for training 3D CNN model
Num_Dis=7
Num_Ite=6

INPUT = np.zeros(((1, Heit, Widt)))
for i in range (Num_Ite):
    for j in range (Num_Sepa):
        TMPIMG = NewInpVid[Num_Dis*j+i,:,:][np.newaxis]
        INPUT = np.append(INPUT, TMPIMG, axis=0)
        # INPUT = torch.cat((INPUT, TMPIMG), 0)
INPUT = INPUT [1:INPUT.shape[0],:,:] # All training datasets are stored in INPUTS continuously
print('  Size of INPUT is ', INPUT.shape, ', Type of INPUT is ', type(INPUT))


#INPUT = transforms.ToTensor()(INPUT[:,:,:])
#INPUT = INPUT.permute(1,2,0) # Original NewInpVid has 3 dimensions
#print('  Size of INPUT after ToTensor is ', INPUT.shape, ', Type of INPUT after ToTensor is ', type(INPUT))
'''
# t4 = time.time()

################### Start to train CNN 3D model #############
print('Frames updating finished,', '3D CNN starts...', '\n')

cuda = torch.device('cuda')
#with torch.cuda.device(0):

#Net3D.load_state_dict(torch.load('model3D.ckpt'))  
Net3D.to(cuda)

LabelS = Read_label('Inputs/Label') # Read the ground truth for actual measured nugget     
# print('Type of Labels is ', type(Labels), ', Size of Labels is ', len(Labels)) # class 'list'. Size is the number of nugget ground truth

#LabelS = LabelS.to(cuda) # send the inputs and targets at every step to the GPU too

Y=60 #For E040-44 96*104; For E68-72: 74*94; For E63-67: 60*78; For E47-49: 104*184; For E75-78: 66*90; For E59-61: 102*118
X=78

Labels = torch.tensor(np.zeros(((len(LabelS), Y, X))), device=cuda)
'''
# For E068-072
for i in range(8):
    Labels[i,:,:] = LabelS[i][0,5:79,9:103]
for i in range(2):
    Labels[i+8,:,:]=LabelS[i+8][0,:,1:95]
'''
#for i in range(10): #For E040-044
#    Labels[i,:,:] = LabelS[i][0,:,:] 

#for i in range(6):# For E047-049
#    Labels[i,:,:]=LabelS[i][0,1:105,1:185]

   
# For E063-067
Lab=4
for i in range(Lab):
    Labels[i,:,:] = LabelS[i][0,:,1:79]
for i in range(6):
    Labels[i+Lab,:,:]=LabelS[i+Lab][0,5:65,1:79]

#for i in range(10): #For E075-76 (81*100 to 66*90)
#    Labels[i,:,:] = LabelS[i][0,7:73,5:95] 

#for i in range(6):
#    Labels[i,:,:] = LabelS[i][0,1:103,1:119]
#for i in range(4):
#    Labels[i+4,:,:]=LabelS[i+4][0,:,:]

# labels shape are [# of considered,Y,X], type is torch.tensor
#Labels = torch.cat(tuple(Labels)) # After this, Labels' size is [# of labels considered,96,104], type is torch.tensor
# print(' After torch.cat, type of Labels is ' , type(Labels), ', size of Labels is ', Labels.shape)

Labels = 255 * Labels 

#Inputs=np.append(Input_13_50_7_45,Input_14_50_7_45,axis=0)
#Inputs=np.append(Inputs,Input_15_50_7_45,axis=0)

# data=np.load('Input_13.npy')

#Inputs=np.load('Input_{:}_NoOtlrs.npy'.format(1)) # Load E040-E044
#for i in range (9): 
#    Inputs=np.append(Inputs,np.load('Input_{:}_NoOtlrs.npy'.format(i+2)),axis=0)

Inputs=np.load('Input_{:}.npy'.format(1)) # Load E068-72
#Inputs=np.load('Input_{:}.npy'.format(1)) # Load E050-E054
#Inputs=np.load('Input_pix_{:}.npy'.format(1))

for i in range (len(LabelS)-1): 
    #Inputs=np.append(Inputs,np.load('Input_pix_{:}.npy'.format(i+2)),axis=0)
    Inputs=np.append(Inputs,np.load('Input_{:}.npy'.format(i+2)),axis=0)
    #Inputs=np.append(Inputs,np.load('Input_{:}.npy'.format(i+2)),axis=0)

#Inputs = Inputs.to(cuda) # send the inputs and targets at every step to the GPU too

INputs= transforms.ToTensor()(Inputs)
INputs = INputs.permute(1,2,0) # shape is [# of frames, height, width]

INputs = INputs.to(cuda)

# print('Type of INputs is ', type(INputs), ', Size of INputs is ',INputs.shape)
# INputs = INputs.unsqueeze(0)
# print('Shape of INputs is ', INputs.shape)

Len_Frm = 30
Num_Dat=int(INputs.shape[0]/Len_Frm) # Total num of datasets
Tag=np.zeros(Num_Dat)
Step=7
Num_Tst=len(LabelS) # Num of datasets left as test datasets
Num_EchDat=35 # Num of datasets for each video

for i in range (int(Num_Dat/Num_EchDat)):# Store each sub-video dataset Tag to identify which nugget ground truth should be used.
    Tag[Num_EchDat*i:Num_EchDat*i+Num_EchDat]=i # 0~(Num_EchDat-1): 0;  Num_EchDat~(2Num_EchDat-1): 1; ...9 

Rodmlst=[] 
Rodmlst=random.sample(range(0,Num_Dat),Num_Dat) # Permute sub-video datasets in a random order. from 0 to Tag.size-1, randomly select Tag.size number of values

for epoch in range (10):
    t1=time.time()
    Pred_acc, Total = 0, 0    
    Running_loss = 0.0
    # print('\n', '3DCNN train epoch: {:} '.format(epoch+1))  
    # F_scores = []
    Pred_True = [] 
    Real_True = []
    Corr_True = []
    
    adjust_lr(optimizer, epoch, lr_decay=15)
    
    for i in range (Num_Dat-Num_Tst):# 135-6=129 For Max. 90-4=86 for Min
        t3=time.time()
        
        print('\n', '3DCNN train epoch: {:} '.format(epoch+1), '  Num of training input: {:}'.format(i+1))
        # print('\n', '  Num of training input: {:}'.format(i+1))
        F_scores = []
        
        Tmp=Rodmlst[i]
        INPUTS = INputs[Tmp*Len_Frm:(Tmp+1)*Len_Frm,:,:]
        INPUTS = INPUTS.unsqueeze(0).unsqueeze(0)
        #print('  Shape of INPUTS is ', INPUTS.shape)
        
        LABEL = Labels[int(Tag[Tmp]),:,:].unsqueeze(0).unsqueeze(0).unsqueeze(0) # Make it have 5 dimensions
        # print('Type of LABEL is ', type(LABEL), ' Shape of LABEL is ', LABEL.shape)        
        print('  NO. sub-video data is ',Tmp, ', NO. Label is ', int(Tag[Tmp]))
        
        #INPUTS = NewInpVid[i*Num_Sepa:Num_Sepa*(i+1),:,:]
        #INPUTS = INPUT[i*Num_Sepa:Num_Sepa*(i+1),:,:]
        
        #INPUTS = INPUTS.unsqueeze(0).unsqueeze(0) # Extend 2 dimensions to INPUTS to make it to be 5 dimensions     
        #print('  Shape of INPUTS is ', INPUTS.shape)
        
        #Labels = Read_label('Data\\Inputs\\Label') # Read the ground truth for actual measured nugget compared with 3D CNN output     
        #Labels = 255 * Labels 
        #print('\n', 'Size of Labels: ', Labels.size(), 'Type of Labels: ', type(Labels)) 
        # [1,96,104], class: torch.Tensor
        
        optimizer.zero_grad()
        #NewInpVid = transforms.ToTensor()(IpVide[0,0,303:391,:,:])# Original starts from 3.
        
        # NewInpVid = transforms.ToTensor()(NewInpVid[:,:,:])
    
        # NewInpVid = NewInpVid.permute(1,2,0) # Original NewInpVid has 3 dimensions
        # print('Shape of NewInpVId = ', NewInpVid.shape, '\n')
              
        # NewInpVid = NewInpVid.unsqueeze(0).unsqueeze(0)      
        # print('Shape of NewInpVId = ', NewInpVid.shape, '\n', 'Type of NewInpVid = ', type(NewInpVid))
        
        Outputs = Net3D(INPUTS.float())
        # Because it is known that there are 391 images input in total
        # Output_size: [1,2,1,96,104]
        
        # Convert input to float is faster regarding running than converting model to double
        
        # Outputs = Net3D(NwFrame[0,0,3:390,:,:])
        _, Predt=torch.max(Outputs, dim=1)
        #print('\n','Size of Predt: ', Predt.size())
        # Predt_size: [1,1,96,104], delete the 2nd dimension of Outputs. Because pick the highest probility of belonging to a class at each pixel
        
        # LABEL=Labels.long() No difference is incurred for labels by adding '.long()' regarding type and size.
        # print('\n', 'Size of LABEL: ', LABEL.size(), 'Type of LABEL: ', type(LABEL)) 

        acc = compute_acc(Predt, LABEL.long())
        Pred_acc += acc
        Total += Y*X
        print('  Average acc:', Pred_acc/Total)
        
        Pred_True, Real_True, Corr_True = compute_F(Predt, LABEL.long())
            # Total evaluated cells are 96*104*batch size
            # print('Pred_True = ', len(Pred_True))
            
        for j in range(2): # Based on the number of classes
            F_scores.append(2*Corr_True[j]/(Pred_True[j]+Real_True[j]))
        print('Pred_outside pixels(0) =', Pred_True[0], ', Pred_onside pixels(0) =', Pred_True[1])
        print('Real_outside pixels(0) =', Real_True[0], ', Real_onside pixels(0) =', Real_True[1])
        print('  Corre_outside pixels(0) =', Corr_True[0], ', Corre_onside pixels(0) =', Corr_True[1])     
        print('  F_scores:', F_scores)
        
        n,c,d,h,w=Outputs.size()# [1,2,1,96,104]
        
        Outputs = Outputs.transpose(1,2).transpose(2,3).transpose(3,4).contiguous().view(-1,c)
        # After, outputs: [9984,2] ?
            
        LABEL = LABEL.view(-1).long() # Make labels to be 1 diamention by multiplying all original diamentions 
        # : [9984]: from (1*1*1*96*104)
        #print(' Max of LABEL is ', np.max(LABEL.numpy()), ', Min of LABEL is ', np.min(LABEL.numpy()))    
        #print('\n','Shape of LABEL is ', LABEL.shape, ', Type of LABEL = ', type(LABEL))#[]
        #print('Latest Output_size = ', Outputs.size())
        # print('After Labels_size = ', Labels.size())
        #print('   Max of Outputs is ', np.max(Outputs.numpy()[:,0]), ', Min of Outputs is ', np.min(Outputs.numpy()[:,0]))
        #print(' Max of Outputs is ', np.max(Outputs.numpy()[:,1]), ', Min of Outputs is ', np.min(Outputs.numpy()[:,1]))
        # np.max(Outputs.numpy()[0,0,0,:,:])
        #print('\n','Shape of Outputs is ', Outputs.shape, ', Type of Outputs = ', type(Outputs))
        #print(Counter(LABEL.numpy().reshape(-1)))

        loss = criterion(Outputs, LABEL) # LABEL shape: [1,1,1,96,104]?
            # outputs: 2 probibilities of being each class at each pixel. labels: actual class of each pixel ?    
        print('Train loss:',loss.item())  
        
        loss.backward() # backpropagate the error (gradients back into the network’s parameters)
        optimizer.step()
            # Does the update
    
            # print statistics
        Running_loss += loss.item() 
        
        # plt.imshow(Predt[0,0,:,:].numpy())
        t4=time.time()
        print('3DCNN precition time for one dataset is {:.4f} sec'.format(t4-t3), end ='\n')
        
    t2 = time.time()
    print('Total training time is {:.4f} sec'.format(t2-t1), end ='\n\n')
     
    torch.save(Net3D.state_dict(), 'model3D.ckpt') 

plt.imshow(Predt[0,0,:,:].numpy())


#############################################################
print('\n','3D CNN Training ends, testing starts...')
t5 = time.time()
output_target = []  

#Net3D.load_state_dict(torch.load('model3D.ckpt'))  

with torch.no_grad():
    Net3D.eval()
    pred_acc, total = 0, 0
    #Dia_X1, Dia_Y1, Flg = 0, 0, 0
    #Dia_X2, Dia_Y2, = 0, 0
    
    #F_scores = []
    Pred_True = [] 
    Real_True = []
    Corr_True = []
    
    for i in range(Num_Tst):# For Min13_14 forming 86 Training datasets. NO. 62, 80, 51 and 41 are used as test datasets
    # For NO.13_14_Cat5_Len30_Stp7_NumEch35: NO. 311, 227, 160, 73, 284, 54, 236, 117, 330, 53 are used for test   
    # For E040-E044 from Min13 to Max14, [38, 156, 53, 55, 52, 84, 185, 166, 138, 226] are left for testing
      # For E040-044 from Min13 to Max14, [257, 105, 158, 280, 64, 240, 82, 177, 162, 120] are not used for training, ExaNor, IniArcht
    # For E068-072, Non-concatenate: Datasets: [330, 112, 303, 316, 101, 96, 349, 302, 127, 265] not used for training.
    # For E063-067, from Min13-Max14, [142, 264, 50, 8, 145, 99, 63, 310, 240, 5] are not used for training
    # For E063-067, from Min13-Max14,[183, 175, 140, 293, 323, 258, 4, 219, 106, 327] are not used for ExaNorm training
    # For E063-067, from Min13-Max14,[72, 38, 253, 349, 63, 338, 130, 230, 110, 116] are not used for NonNorm training
    # For E063-067, from Min13-Max14,[265, 202, 310, 148, 249, 24, 145, 169, 119, 7] are not used for ExaNorm training, Archt2
    # For E047-049, from Target13-Max14, [12, 62, 80, 48, 1, 28] are not used for ExaNorm training, Atcht3
      # For E047-049, from Target13-Max14, [171, 100, 57, 155, 114, 187] are not used for ExaNorm training, InitArcht
    # For E075-078, from Low13-Max14, [135, 84, 91, 46, 272, 95, 158, 113] are not used for ExaNorm Training, Archt3
    # For E075-078, from Low13-Max14, [64, 54, 53, 222, 274, 94, 184, 46] are not used for ExaNorm Training, InitModel
    # For E059-061, from Target13-Max14: [77, 89, 68, 35, 78, 172] are not used for ExaNorm Training, InitModel
        print('\n','  Num of test input: {:}'.format(i+1))
        
        t7 = time.time()
        F_scores = []
        
        Tmp=Rodmlst[Num_Dat-i-1]
        
        INPUTS = INputs[Tmp*Len_Frm:(Tmp+1)*Len_Frm,:,:]
        INPUTS = INPUTS.unsqueeze(0).unsqueeze(0)
        #print('  Shape of INPUTS is ', INPUTS.shape)# [1,1,50(length of a sub-video data),96,104]
        
        LABEL = Labels[int(Tag[Tmp]),:,:].unsqueeze(0).unsqueeze(0).unsqueeze(0) # Make it have 5 dimensions
        print('  NO. sub-video data for Test is ',Tmp, ', NO. Label is ', int(Tag[Tmp]))
        
        Outputs = Net3D(INPUTS.float())
        #print('Shape of Outputs is ', Outputs.shape)#[1,2,1,96,104]
        _, Predt=torch.max(Outputs, dim=1)
        output_target.append(Predt)
        # print('   Type of output_target is ', type(output_target))# class 'list'
        # print('output_target_size = ', len(output_target))# size is 1
        # for i in rang(2)
        # print(' '.join('%5s' % classes[pred[j,]] for j in range(2)))            
        acc = compute_acc(Predt, LABEL.long())
        pred_acc += acc
        total += Y*X
        print('Average acc:', pred_acc/total)
            
        Pred_True, Real_True, Corr_True = compute_F(Predt, LABEL.long())
        # Total evaluated cells are 96*104*batch size
        # print('Pred_True = ', len(Pred_True))
        
        for j in range(2): # Based on the number of classes
            F_scores.append(2*Corr_True[j]/(Pred_True[j]+Real_True[j]))
        print(' Pred_outside=', Pred_True[0], ', Pred_onside=', Pred_True[1])
        print(' Real_outside=', Real_True[0], ', Real_onside=', Real_True[1])
        print('  Corre_outside=', Corr_True[0], ', Corre_onside=', Corr_True[1])
        print(' F_scores:', F_scores)
        
        # print('input_size = ', input.size())
        # print('orig output_size = ', outputs.size())
        # print('orig labels_size = ', labels.size())
        
        n,c,d,h,w=Outputs.size()# [1,2,1,96,104]
        Outputs = Outputs.transpose(1,2).transpose(2,3).transpose(3,4).contiguous().view(-1,c)
        # After, Outputs size is [9984,2] 
            
        LABEL = LABEL.view(-1).long() # Make labels to be 1 diamention by multiplying all original diamentions 
        
        # print('Latest Outputs_size = ', Outputs.size())# [9984,2]
        # print('After LABEL_size = ', LABEL.size()) # [9984]
            
        loss = criterion(Outputs, LABEL) # Outputs [1,2,1,96,104]. LABEL [1,1,1,96,104]!
        print('Test loss:',loss.item())
        t8 = time.time()
        print('3DCNN prediction time for one dataset is {:.4f} sec'.format(t8-t7), end ='\n\n')
        
    t6 = time.time()
    print('Total testing time is {:.4f} sec'.format(t6-t5))

np.save('Exp3DTrndat.npy', Rodmlst[Num_Dat-Num_Tst:Num_Dat])
print('Unused training data is:', Rodmlst[Num_Dat-Num_Tst:Num_Dat])    
plt.imshow(Predt[0,0,:,:])  
output_target = torch.cat(tuple(output_target))
print('Test_output', output_target.shape) # [3(# of test data),1,96,104]
#print('   Type of output_target is ', type(output_target))# class 'torch.Tensor'
    # output_target is accumulated over all test samples
