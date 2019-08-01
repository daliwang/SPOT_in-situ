# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:35:12 2019

@author: Jian Zhou
"""

import torch
import torch.optim as optim
from Load_inputs import Dataloder# Load dataset for 3D CNN inputs
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import Counter
#from Model3D import NET
from Model3DTest import NET # InitArcht
#from CNN3D_archt import NET # NonConcatenate architecture
#from CNN3D_archt2 import NET
#from CNN3D_archt3 import NET
import torchvision
from torchvision import transforms

from PIL import Image
import os
import os.path
import random

os.environ["CUDA_VISIBLE_DECIVES"]="0"

cuda=torch.device("cuda")

Net3D = NET()
Net3D.to(cuda)

#print(Net3D)
'''
dataloader = Dataloder()
inputloader, Num_input = dataloader.getloader()
# Num_input: Total number of input images
print('Num of input images = ', Num_input, '\n')

# classes, trainloader, testloader = dataloader.getloader() #, num_inputs = dataloader.getloader()
# print('testloader_batch_size = ', testloader.batch_size)
# print('num of testloader = ', testloader.__len__())
# print('num of testloader = ', num_inputs)
'''
criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.3,1.0]).cuda()) #0.005,1.0
#combines nn.LogSoftmax() and nn.NLLLoss()

optimizer = optim.SGD(Net3D.parameters(), lr=0.01, momentum=0.9)
#Stochastic Gradient Descent simplest update weights   

def adjust_lr(optimizer, epoch, lr_decay=15):
    lr = 0.01 *(0.1**((epoch)//lr_decay))
    if (epoch) % lr_decay == 0:
        print('LR is set to: ', lr)
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def compute_acc(pred, label): # calculate accuracy
    pred = pred.unsqueeze(1)
    #label = label.unsqueeze(1)# Extend 1 dimension because it has one less dimension compared with pred!    
    # print('pred size for accy = ', pred.size())
   
    # print((label == 2).sum())
    #print('pred size:', pred.shape) #[1,1,96,104]
    #print('label size:', label.shape) #[1,1,96,104]
    # print(' '.join('%5s' % classes[pred[j]] for j in range(2)))
    
    acc_sum = (pred == label).sum()
    # Totally 96*104*8 predictions/comparisons 
    return acc_sum.item()
    
def compute_F(pred, label): # calculate F1 scores for 2 labels
    pred = pred.unsqueeze(1).numpy()
    label = label.numpy()
    #pred = pred.numpy()
    
    #print(' pred size = ', pred.shape) # Make them in the same size
    #print(' label size = ', label.shape)
    
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
    Labels = []
    _Label=os.path.join(labeldir,'GrdTruh.png')
    
    _Labels = Image.open(_Label)
    _Labels = transforms.ToTensor()(_Labels)
    Labels.append(_Labels)
    
    return Labels

'''
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
        # print('Shape of _Labels is ', _Labels.shape)# [1,96,104]

        Labels.append(_Labels) # Labels' type is <class 'list'>       
    
    return Labels
'''  
###################    
'''
    LABELS = []
    
    _label=os.path.join(labeldir,'GrdTruh.png')
    
    LABELS.append(_label)# class 'list'
    # print('shape of _label is ', np.shape(LABELS)) # (1,)

    _Labels = Image.open(LABELS[0]) # Max is 1, Min is 0
    # Class list cannot use read 
    #_Labels=np.asarray(_Labels)
    #print('   Type of _Labels is ', type(_Labels), ', Shape is ', _Labels.shape)
    #print('Max of LABELS = ', np.max(_Labels), ', Min = ', np.min(_Labels))
    # print('shape of LABELS is ', LABELS.shape) # [96,104]!!
    
    # print('   Type of _Labels is ', type(_Labels))# <class 'PIL.PngImagePlugin.PngImageFile'>!

    _Labels = transforms.ToTensor()(_Labels) # It extends 1 dimension to _Labels!! [1,96,104]   
    # print('After ToTensor, ', 'Max of LABELS = ', np.max(LABELS), ', Min = ', np.min(LABELS))
    # print('Shape of LABELS is ', LABELS.shape, ', Type of LABELS is ', type(LABELS),'\n')
    
    # transforms.ToTensor divide elements in _Labesl(<class 'PIL.PngImagePlugin.PngImageFile'>) by 255. 
    # transforms.ToTensor add a dimension to make _Labels to be 3 dimensions: [1,96,104] 
    
    #print('Shape of _Labels is ', _Labels.shape, 'Type of _Labels is ', type(_Labels), 'Max =', np.max(_Labels.numpy()),'Min = ', np.min(_Labels.numpy()))
        
    # transforms.ToTensor() only works for numpy with 2 dimensons or 3 dimensons 
    return _Labels
'''

############################# Main function #######################

print('Test session', '\n')
t3 = time.time()
output_target = []  

Net3D.load_state_dict(torch.load('model3D.ckpt'))  
  
# Num_Itera= 0 # Count total iterations over batches

# InputImags = np.zeros(((Num_input, 96, 104)))
# InputImags = []

#LabelS = Read_label('Data\\Inputs\\Label') # Read the ground truth for actual measured nugget     
LabelS = Read_label('Ground_Truth/Target/E065/000015/GrdTrh_relocated') # Read the ground truth for actual measured nugget 
#print('Type of LabelS is ', type(LabelS), ', Size of LabelS is ', len(LabelS)) # class 'list'. Size is the number of nugget ground truth

Y=60 # For E040-44 96*104; For E50-54, 92*104; For E68-E72: 74*94. For E63-67: 60*78; For E47-49: 104*184; For E75-78: 66*90; For E59-61: 102*118
X=78

#Labels = torch.tensor(np.zeros(((len(LabelS), Y, X))))
Labels=torch.zeros([len(LabelS), Y, X], device=cuda, dtype=torch.float32)# Set data type because it will be float64 which 
#print(' Type of Labels is ' , type(Labels), ', size of Labels is ', Labels.shape)

#for i in range(len(LabelS)): #For 105*185 to 104*184 (H*W)
#    Labels[i,:,:] = LabelS[i][0,1:105,1:185]

# For E75-78
#for i in range(len(LabelS)): #For 81*100 to 66*90 (H*W)
#    Labels[i,:,:] = LabelS[i][0,7:73,5:95]

#for i in range(len(LabelS)): #For 104*184 (H*W)
#    Labels[i,:,:] = LabelS[i][0,1:105,1:185]

# For E068-072
#for i in range(len(LabelS)): #For 84*112 (H*W)
#   Labels[i,:,:] = LabelS[i][0,5:79,9:103]
#for i in range(len(LabelS)):# For 74*95 (H*W)
#    Labels[i,:,:]=LabelS[i][0,:,1:95]

# For E063-67
for i in range(len(LabelS)): #For 70*79 (H*W). (Max, High, Target)
    Labels[i,:,:] = LabelS[i][0,5:65,1:79]
#for i in range(len(LabelS)):# For 60*79 (H*W)
#    Labels[i,:,:]=LabelS[i][0,:,1:79]

#for i in range(len(LabelS)):
#    Labels[i,:,:] = LabelS[i][0,:,:] # For 96*104 or 92*104
    
#for i in range(len(LabelS)):
#    Labels[i,:,:] = LabelS[i][0,1:103,1:119] # For 102*118
    #Labels[i,:,:] = LabelS[i][0,:,:]
    #Labels[i,:,:] = LabelS[i][0,1:105,1:185]
#for i in range(len(LabelS)): # For 70*79
#    Labels[i,:,:]=LabelS[i][0,5:65,1:79]  
    
# labels shape are [# of considered,Y,X], type is torch.tensor
#Labels = torch.cat(tuple(Labels)) # After this, Labels' size is [# of labels considered,96,104], type is torch.tensor
# print(' After torch.cat, type of Labels is ' , type(Labels), ', size of Labels is ', Labels.shape)

Labels = 255 * Labels 

'''
Labels = Read_label('Data\\Inputs\\Label') # Read the ground truth for actual measured nugget     
#print('Type of Labels is ', type(Labels), ', Size of Labels is ', len(Labels)) # class 'list'. Size is the number of considered nugget ground truth
Labels = torch.cat(tuple(Labels)) # After this, Labels' size is [Num of label considered,96,104], type is torch.tensor
#print(' After torch.cat, type of Labels is ' , type(Labels), ', size of Labels is ', Labels.shape)

Labels = 255 * Labels 

#Area_GT=np.count_nonzero(Labels==1)
'''
X_ra=5.2 # 8.8,6.1 For E040-44; 8.14,5.23 For E050-54; 8.01,5.83 For E068-072; 5.2,3.6 for E63-67; 9.32, 6.63 for E47-49; 10,7 for E075-78. 3.8,2.5 for E059-61.
Y_ra=3.6

Flg = 0
Dia_GT=np.zeros(2)

for j in range(X):
    if Flg ==0 and 1 in Labels[0,:,j]:
        Lef=j
        Flg=1
    if Flg ==1 and not(1 in Labels[0,:,j]):
        Rig=j-1
        break
Dia_GT[0]=Rig-Lef+1 # Measured diameter (pixel) in horizontal axis

Flg = 0    
for j in range(Y):
    if Flg == 0 and 1 in Labels[0,j,:]:
        Up = j
        Flg = 1
    if Flg == 1 and not(1 in Labels[0,j,:]):
        Dow = j-1
        break        
Dia_GT[1]=Dow-Up+1 # Vertical diameter (pixel) based on Label ground truth

print('Actual measured horizontal diameter in pixel is', Dia_GT[0],'\n')

Dia_GTH=(Dia_GT[0]/X_ra+Dia_GT[1]/Y_ra)/2
print('Actual measured diameter average over Horizontal and Vertical in mm is', Dia_GTH,'\n')

#Inputs=np.load('Input_Max13_50_7_35_NoOtlrs.npy') # E040-E044
#Inputs=np.load('Input_2_ExaNor.npy')
Inputs=np.load('Input_Target15_30_7_35_E65_ExaNor.npy')
#Inputs=np.load('Input_Min15_30_7_35_NoOtlrs.npy')
#Inputs=np.load('Input_pix_E043_High15.npy') # E040-E044
#Nam='Input_pix_E043_High15'

#Inputs=Input_Tagt15_50_7_35_NoOtlrs # Every time initilize Inputs!
#Inputs=np.append(Inputs,Input_15,axis=0)

INputs= transforms.ToTensor()(Inputs)
INputs = INputs.permute(1,2,0) # shape is [# of considered selected frames, height, width]

INputs=INputs.to(cuda)
# print('Type of INputs is ', type(INputs), ', Size of INputs is ',INputs.shape) # torch.Tensor
# INputs = INputs.unsqueeze(0)
# print('Shape of INputs is ', INputs.shape)

Len_Frm = 30
Num_Dat=int(INputs.shape[0]/Len_Frm)
Tag=np.zeros(Num_Dat)
Step=7
#Num_Tst=3
# Num_EchDat=30, instead by Num_Dat!!

for i in range (Labels.shape[0]):# Store each sub-video dataset Tag to identify which nugget ground truth should be used.
    Tag[Num_Dat*i:Num_Dat*i+Num_Dat]=i    

Rodmlst=[] 
Rodmlst=random.sample(range(0,Num_Dat),Num_Dat) # Permute sub-video datasets in a random order. from 0 to Tag.size-1, randomly select Tag.size number of values
 
#Pre_Nug=zeros

with torch.no_grad():
    Net3D.eval()
    pred_acc, total = 0, 0
    #Dia_X1, Dia_Y1, Flg = 0, 0, 0
    #Dia_X2, Dia_Y2, = 0, 0
    
    #F_scores = []
    Pred_True = [] 
    Real_True = []
    Corr_True = []
    '''
    for i, data in enumerate(inputloader, 0):# Total 20, batch size is 8, so i from 0 to 19
        
        # print('Acutal i = ', i)
        # Num_Itera=Num_Itera+1 # Count current iteration of used batches
        
        # print('Num of iteration = ', Num_Itera)
        
        # F_scores = []
        #Ave_Diam = 0
        #Dia = []
        inputs = data# type of inputs is <class 'torch.Tensor'>
        
        #labels = labels*255
        #[Heig,Widt]=inputs.shape()
        InputImags[i,:,:] = inputs[0,0,:,:]
        # nputImags.append(inputs[0,0,:,:)
    
    # print('Max of InputImags = ', np.max(InputImags[:,:,:]), ', Min of InputImags = ', np.min(InputImags[:,:,:]))
    # Max is 37.53, Min is -12.72. After normalization in load_data, it is changed from [0,1]!!
    # print('Type of InputImags is ', type(InputImags)) #<class 'numpy.ndarray'>. Because it is initialized as numpy.

    IpVide = InputImags[np.newaxis][np.newaxis] # 5 dimensions
    
    NewIpVide=IpVide[0,0,8:Num_input,:,:]# Upon CNN2D: T_0=8, T_olier=9 for Max_15; T_0=8 for Max_14; T_0=10 for Max_13 
    # T_0=24 for Min_13, T_0=26 and 34 outliers for Min_14, T_0=22 and 25 outliers for Min_15 
    print('Shape of NewIpVide is ', NewIpVide.shape,', Type of NewIpVide is ', type(NewIpVide))

# for i in range (Num_input-T_0):
#    NewInpVid[i,:,:] = IpVide[0,0,i,:,:]
    
    # NewInpVid=IpVide[0,0,:,:,:]
    
    Num_Sepa=50 # Continuous 50 frames are used as a input for training 3D CNN model
    Num_Dis=7  # Steps for selecting frame (Every n steps to pick a frame) to assumbly a subdataset
    Num_Tst=7 # Which subdataset is used for test
    
    INPUT = np.zeros(((1, 96, 104)))
    for i in range (7):
        for j in range (Num_Sepa):
            TMPIMG = NewIpVide[Num_Dis*j+i,:,:][np.newaxis]
            INPUT = np.append(INPUT, TMPIMG, axis=0)
            # INPUT = torch.cat((INPUT, TMPIMG), 0)
    INPUT = INPUT [1:INPUT.shape[0],:,:] # All training datasets are stored in INPUTS continuously
    print('  Size of INPUT is ', INPUT.shape, ', Type of INPUT is ', type(INPUT))
    
    INPUT = transforms.ToTensor()(INPUT[:,:,:])
    INPUT = INPUT.permute(1,2,0) # Original NewInpVid has 3 dimensions
    print('  Size of INPUT after ToTensor is ', INPUT.shape, ', Type of INPUT after ToTensor is ', type(INPUT))   
    '''  
    
    #for i in range(Num_Sepa):
    #    NewInpVid[Num_Sepa-1-i,:,:] = IpVide[0,0,Num_input-1-i,:,:]

    #print('Init shape of NewInpVId = ', NewInpVid.shape) # [80,96,104]
    # inputs_size: [batch size, # of input channels(image value of each pixel), # of rows, # of columns]
    # labels.size()) [batchsize,1,96,104]
    '''
    print('  Before transforms.ToTensor: ')
    print('Type of NewInpVid is ', type(NewInpVid), ', Max of NewInpVid = ', np.max(NewInpVid), ', Min = ', np.min(NewInpVid))
    # class 'numpy.ndarray'
    
    NewInpVid = transforms.ToTensor()(NewInpVid)
    # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    print('\n', '   After ToTensor:')
    print('Type of NewInpVid is ', type(NewInpVid), ', Max of NewInpVid = ', np.max(NewInpVid.numpy()), ', Min = ', np.min(NewInpVid.numpy()))
    # class 'torch.Tensor' 
    # Max is still 37.53, Min is -12.72. The range doesn't changed from before (InputImags). While dimensions are exchanged [104,80,96] !
    # print('Orig shape of NewInpVId = ', NewInpVid.shape) 
    
    NewInpVid = NewInpVid.permute(1,2,0) # Original NewInpVid has 3 dimensions
    #print('Temp shape of NewInpVId = ', NewInpVid.shape) # [80,96,104]
          
    NewInpVid = NewInpVid.unsqueeze(0).unsqueeze(0)      
    #print('Latest shape of NewInpVId = ', NewInpVid.shape, ', Type of NewInpVid = ', type(NewInpVid))
    '''
    '''
    NewInpVid=INPUT[(Num_Tst-1)*Num_Sepa:INPUT.shape[0],:,:]
    NewInpVid = NewInpVid.unsqueeze(0).unsqueeze(0)      
    #print('Latest shape of NewInpVId = ', NewInpVid.shape, ', Type of NewInpVid = ', type(NewInpVid))

    Labels = Read_label('Data\\Inputs\\Label')
    # print('Max of labels = ', np.max(Labels.numpy()[:,:,:]), ', Min of a batch of frames = ', np.min(Labels.numpy()[:,:,:], '\n\n'))
    # Range is [0, 1/255]
    Labels = 255 * Labels # Upgrade Labels' range to be [0,1]. Class is totensor  
    print('\n', 'Size of Labels: ', Labels.size(), 'Type of Labels: ', type(Labels)) 
    '''
    
    #ww=NewInpVid.float()
    #print('Max = ', np.max(ww.numpy()), ', Min = ', np.min(ww.numpy(), '\n\n'))
    
    F_Scores=np.zeros(((Num_Dat,1,2)))
    CORR_True=np.zeros(((Num_Dat,1,2)))
    Ave_Fscores=np.zeros((1,2))
    Ave_Corre_True=np.zeros((1,2))
    
    AreaNgt=np.zeros(Num_Dat)
    
    #Flg= 0
    #Dia_T=np.zeros(Num_Dat)
    
    for i in range(Num_Dat):
    #for i in range (1):    
    
        '''
        for i in range(Len_Frm):    
            for j in range(104):
                if Flg == 0 and 1 in output_target[i,0,:,j]:
                    #if Flg ==0 and output_target[i,0,:,j] and 1 in Dia_GT/2:
                    Tp = (output_target[i,0,:,j]==1).nonzero() # Index of value 1 in the column
                    for k in range (Tp.shape[0]):
                        if not(0 in output_target[i,0,Tp[k],j:j+round(Dia_GT/2)][0,:]):
                            Lef=j
                            Flg=1
                            break
                elif Flg ==1 and not(1 in output_target[i,0,:,j]):
                    Rig=j-1
                    Flg=0
                    break
            Dia_T[i]=Rig-Lef
        #Diat=set(Dia)    
        Diat=[x/8.14 for x in Dia]
        Diat=np.round(Diat,decimals=2)
        Diat_mm=list(set(Diat))
        '''
        
        print('\n','  Num of test input: {:}'.format(i+1))
        F_scores = [] 
        
        Tmp=Rodmlst[i]
        
        INPUTS = INputs[Tmp*Len_Frm:(Tmp+1)*Len_Frm,:,:]
        INPUTS = INPUTS.unsqueeze(0).unsqueeze(0)
        #print('  Shape of INPUTS is ', INPUTS.shape)# [1,1,(length of a sub-video data),96,104]
        
        LABEL = Labels[int(Tag[Tmp]),:,:].unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        print('  NO. sub-video data for test is ',Tmp, ', NO. Label is ', int(Tag[Tmp]))
        '''
        INPUTS = INputs[(i+14)*Len_Frm:(i+1+14)*Len_Frm,:,:]
        INPUTS = INPUTS.unsqueeze(0).unsqueeze(0)
        
        LABEL = Labels[2,:,:].unsqueeze(0) # 3 dimensions
        '''
        # print('  NO. sub-video data is ',Tmp, ', NO. Label is ', int(Tag[Tmp]))
        #print('Type of LABEL is ', type(LABEL), ' Shape of LABEL is ', LABEL.shape)#torch.Tensor; [1,96,104]
        t1=time.time()       
        '''
        print('Shape of INPUTS is ', INPUTS.shape, ', Type of INPUTS is ', type(INPUTS))#[1,1,(length of a sub-video data),96,104],torch.Tensor
        print('  NO.1 pixel value in INPUTframe1 is ', INPUTS.numpy()[0,0,29,15,5], ', NO.2 pixel value in INPUTframe1 is ', INPUTS.numpy()[0,0,29,50,60])
        print(' NO.1 pixel value in INPUTframe2 is ', INPUTS.numpy()[0,0,28,15,5], ', NO.2 pixel value in INPUTframe2 is ', INPUTS.numpy()[0,0,28,50,60])
        print('  Max of INPUTframe1 is ', np.max(INPUTS.numpy()[0,0,29,:,:]), ', Min of INPUTframe1 is ', np.min(INPUTS.numpy()[0,0,29,:,:]))
        print(' Max of INPUTframe2 is ', np.max(INPUTS.numpy()[0,0,28,:,:]), ', Min of INPUTframe2 is ', np.min(INPUTS.numpy()[0,0,28,:,:]))
        '''
        Outputs = Net3D(INPUTS.float())
        '''       
        print('\n','Shape of Outputs is ', Outputs.shape, ', Type of outputs = ', type(Outputs))#[1,2,1,96,104], torch.tensor
        print(' Max of Pag1 Outputs is ', np.max(Outputs.numpy()[0,0,0,:,:]), ', Min of Pag1 Outputs is ', np.min(Outputs.numpy()[0,0,0,:,:]))
        print(' Max of Pag2 Outputs is ', np.max(Outputs.numpy()[0,1,0,:,:]), ', Min of Pag2 Outputs is ', np.min(Outputs.numpy()[0,1,0,:,:]),'\n')       
        print(' NO.1 pixel value in Pag1Out is ', Outputs.numpy()[0,0,0,15,5], ', NO.1 pixel value in Pag2Out is ', Outputs.numpy()[0,1,0,15,5])
        print(' NO.2 pixel value in Pag1Out is ', Outputs.numpy()[0,0,0,50,60], ', NO.2 pixel value in Pag2Out is ', Outputs.numpy()[0,1,0,50,60])
        '''
        _, PRedt=torch.max(Outputs, dim=1)
        '''
        print('Shape of Predt is ', Predt.shape, 'Type of Predt = ', type(Predt))#[1,1,92,104], 'torch.Tensor'
        print('   Final class of NO.1 pixel is ', Predt.numpy()[0,0,15,5],',   Final class of NO.2 pixel is ', Predt.numpy()[0,0,50,60])   
        #print('  Max of Predt = ', np.max(Predt.numpy()[0,0,:,:]), ', Min of Predt = ', np.min(Predt.numpy()[0,0,:,:]), '\n') # Max is 1, Min is 0
        '''
        
        t2=time.time()
        print('Time just for CNN 3D prediction is {:.4f} sec'.format(t2-t1))
        
        # print('type of labels is ', type(labels))
        # print('type of inputs is ', type(inputs))
        
        # outputs = Net3D(inputs)
        #_, pred = torch.max(outputs, dim=1)
        
         # Outputs_size: [batch size, # of classes(Probility of being each class), # of rows, # of columns]!
        # pred_size: [batchsize,96,104], Pick the highest probility of belonging to a class at each pixel
        
        #print('Type of inputs = ', type(inputs), ', Shape of inputs = ', inputs.shape)
        #print('Type of outputs = ', type(Outputs))
        # print('Predt size = ', Predt.size(), ', Type of Predt is ', type(Predt)) # class 'torch.Tensor',[1,1,96,104]
        
        #print('Type of pred.numpy', type(pred.numpy()))
        # class 'numpy.ndarray'
        
        #PRED=pred.numpy()
        #print('A frame center = ', PRED[0,47,51], ', Left up of a frame = ', PRED[0,0,0]) 
        #print('Max of a batch of frames = ', np.max(pred.numpy()[:,:,:]), ', Min of a batch of frames = ', np.min(pred.numpy()[:,:,:]))
        
        #print('\n', 'Any prediction larger than 1: ', any(PRED[0,0,:]>1))
        #print('Any prediction smaller than 0: ', any(PRED[0,0,:]<0))
        # print('Any prediction between 0 and 1: ', any(PRED[0,0,:]>1))
        # for j in range (104):
        #    if PRED[0,0,j]<1 and PRED[0,0,j]>0:
        #        print('outliner')

        #Pred=pred.numpy()
        # print('pred row = ', Pred.shape[0], ', pred_column = ', Pred.shape[1])
        # print('labels size = ', labels.size())
        
        # output_target.append(Predt)
        # print('   Type of output_target is ', type(output_target))# class 'list'
        # pred store to output_target
        # pred_size is [8(batch size),96,104]
        # labels_size is [8,1,96,104]
        # print('output_target_size = ', len(output_target))# size is 1
        # for i in rang(2)
        # print(' '.join('%5s' % classes[pred[j,]] for j in range(2)))            
        acc = compute_acc(PRedt, LABEL.long())
        pred_acc += acc
        total += Y*X
        print('Average acc:', pred_acc/total)
          
        Predt=PRedt.to('cpu')
        LABEL_cpu=LABEL.to('cpu')

        output_target.append(Predt)
        
        Pred_True, Real_True, Corr_True = compute_F(Predt, LABEL_cpu.long())
        # Total evaluated cells are 96*104*batch size
        # print('Pred_True = ', len(Pred_True))
        
        for j in range(2): # Based on the number of classes
            F_scores.append(2*Corr_True[j]/(Pred_True[j]+Real_True[j]))
        print(' Pred_outside=', Pred_True[0], ', Pred_onside=', Pred_True[1])
        print(' Real_outside=', Real_True[0], ', Real_onside=', Real_True[1])
        print('  Corre_outside=', Corr_True[0], ', Corre_onside=', Corr_True[1])
        print(' F_scores:', F_scores)
        
        F_Scores[i,0,:]=F_scores
        CORR_True[i,0,:]=Corr_True
        # print('input_size = ', input.size())
        # print('orig output_size = ', outputs.size())
        # print('orig labels_size = ', labels.size())
        
        n,c,d,h,w=Outputs.size()# [1,2,1,96,104]
        Outputs = Outputs.transpose(1,2).transpose(2,3).transpose(3,4).contiguous().view(-1,c)
        # After, Outputs size is [9984,2] 
        #print('   Max of Outputs is ', np.max(Outputs.numpy()[:,0]), ', Min of Outputs is ', np.min(Outputs.numpy()[:,0]))
        #print(' Max of Outputs is ', np.max(Outputs.numpy()[:,1]), ', Min of Outputs is ', np.min(Outputs.numpy()[:,1]))
            
        LABEL = LABEL.view(-1).long() # Make labels to be 1 diamention by multiplying all original diamentions 
        # Original labels.size(): [10,1,96,104]
        # Then:[99840]: 10*96*104
        #print('   Max of LABEL is ', np.max(LABEL.numpy()), ', Min of LABEL is ', np.min(LABEL.numpy()))
        
        # print('Latest Outputs_size = ', Outputs.size())# [9984,2]
        # print('After LABEL_size = ', LABEL.size()) # [9984]
            
        loss = criterion(Outputs, LABEL) # Check both shape??!!
            # labels: [79872]: (8*1*96*104)
        print('Test loss:',loss.item(), end ='\n\n')
        
        AreaNgt[i]=(np.count_nonzero(Predt[0,0,:,:]==1))
        
    t4 = time.time()
    print('Profiling time of prediction is {:.4f} sec'.format(t4-t3))
    Profiling_time = t4-t3    
    
    Ave_Fscores[0,0]=np.mean(F_Scores[:,0,0])
    Ave_Fscores[0,1]=np.mean(F_Scores[:,0,1])
    Ave_Corre_True[0,1]=np.mean(CORR_True[:,0,1])
    Ave_Corre_True[0,0]=np.mean(CORR_True[:,0,0])
        
plt.imshow(Predt[0,0,:,:])  
output_target = torch.cat(tuple(output_target))

Flg= 0
Dia=np.zeros((Num_Dat,2))# Predicted diamters based on all datasets. 1st column in horizontal, 2nd column in vertical

for i in range(Num_Dat):# NO.0-34
####### Find Diameter in pixel in horizontal axis   
    Lef, Rig = 0, 0
    Up, Dow = 0, 0

    for j in range(X):
        if Flg == 0 and 1 in output_target[i,0,:,j]:
        #if Flg ==0 and output_target[i,0,:,j] and 1 in Dia_GT/2:
            Tp = (output_target[i,0,:,j]==1).nonzero() # Index of value 1 in the column
            for k in range (Tp.shape[0]):
                if not(0 in output_target[i,0,Tp[k],j:j+int(Dia_GT[0]/2)][0,:]):
                    if not(0 in output_target[i,0,max(0,int(Tp[k]-Dia_GT[1]/2)):Tp[k],j+int(Dia_GT[0]/2)]) or not(0 in output_target[i,0,Tp[k]:min(Y,int(Tp[k]+Dia_GT[1]/2)),j+int(Dia_GT[0]/2)]):
                        Lef=j
                        Flg=1
                        break
        #elif Flg ==1 and not(1 in output_target[i,0,:,j]):
        elif Flg ==1 and not(1 in output_target[i,0,max(0,Tp[k]-int(Dia_GT[1]/2)):min(Y,Tp[k]+int(Dia_GT[1]/2)),j]):
            Rig=j-1
            Flg=0
            break
    Dia[i][0]=Rig - Lef + 1
    
# Fing Diameter in pixel in vertical axis    
    Flg = 0
    for j in range(Y):
        if Flg == 0 and 1 in output_target[i,0,j,:]:
        #if Flg ==0 and output_target[i,0,:,j] and 1 in Dia_GT/2:
            Tp = (output_target[i,0,j,:]==1).nonzero() # Index of value 1 in the column
            for k in range (Tp.shape[0]):
                if not(0 in output_target[i,0,j:j+int(Dia_GT[1]/2),Tp[k]][:,0]):
                    #if not(0 in output_target(i,0,j+int(Dia_GT[1]/2),Tp[k]:Tp[k]+int(Dia_GT[0]/2))) or not(0 in output_target(i,0,j+int(Dia_GT[1]/2),Tp[k]-int(Dia_GT[0]/2):Tp[k])): 
                    if not(0 in output_target[i,0,int(j+Dia_GT[1]/2),Tp[k]:min(int(Tp[k]+Dia_GT[0]/2),X)]) or not(0 in output_target[i,0,int(j+Dia_GT[1]/2),max(0,int(Tp[k]-Dia_GT[0]/2)):Tp[k]]):
                        Up = j
                        Flg = 1
                        break
        elif Flg == 1 and not(1 in output_target[i,0,j,max(Tp[k]-int(Dia_GT[0]/2),0):min(Tp[k]+int(Dia_GT[0]/2),X)]):
            Dow = j-1
            Flg = 0
            break
    Dia[i][1]=Dow - Up +1
#Diat=set(Dia)

DiaT=np.zeros(Num_Dat)
    
DiatX=[x/X_ra for x in Dia[:,0]]
DiatY=[y/Y_ra for y in Dia[:,1]]
for i in range(Num_Dat):
    DiaT[i]=(DiatX[i]+DiatY[i])/2

DiaT=np.round(DiaT,decimals=2)
Diat_mm=list(set(DiaT))
AveDiat=np.mean(Diat_mm)
AveArea=np.mean(AreaNgt)

#np.save('Out_target.npy', output_target)
print('Predicted diameters are', Diat_mm)
print('Test_output', output_target.shape) # [(# of test data),1,96,104]
# print('   Type of output_target is ', type(output_target))# class 'torch.Tensor'
# output_target is accumulated over all test samples
