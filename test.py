# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:57:00 2019

@author: Jian Zhou
"""
import torch
import torch.optim as optim
from load_data import Dataloder
from model import Net
import torch.nn as nn
import numpy as np
import time
from collections import Counter
import matplotlib.pyplot as plt


net = Net()
# print(net)
dataloader = Dataloder()

classes, trainloader, testloader = dataloader.getloader() #, num_inputs = dataloader.getloader()

print('testloader_batch_size = ', testloader.batch_size)
print('num of testloader = ', testloader.__len__())
# print('num of testloader = ', num_inputs)

criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.01,1.0])) #0.005,1.0
#combines nn.LogSoftmax() and nn.NLLLoss()

#optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
#Stochastic Gradient Descent simplest update weights   

def adjust_lr(optimizer, epoch, lr_decay=10):
    lr = 0.01 *(0.1**((epoch-1)//lr_decay))
    if (epoch-1) % lr_decay == 0:
        print('LR is set to: ', lr)
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def compute_acc(pred, label): # calculate accuracy
    pred = pred.unsqueeze(1)
         
    # print('pred size for accy = ', pred.size())
   
    # print((label == 2).sum())
    #print('pred size:', pred.shape)
    #print('label size:', label.shape)
    # print(' '.join('%5s' % classes[pred[j]] for j in range(2)))
    
    acc_sum = (pred == label).sum()
    # Totally 96*104*8 predictions/comparisons 
    return acc_sum.item()
    
def compute_F(pred, label): # calculate F1 scores for 2 labels
    pred = pred.unsqueeze(1).numpy()
    label = label.numpy()
    
    # print('pred size = ', pred.np.size())
    # print('labels size = ', labels.np.size())
    
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





print('Test session')
t3 = time.time()
output_target = []  
net.load_state_dict(torch.load('model.ckpt'))  
  
Num_Itera= 0 # Count total iterations over batches
  
with torch.no_grad():
    net.eval()
    pred_acc, total = 0, 0
    
    Pred_True = [] 
    Real_True = []
    Corr_True = []
    
    for i, data in enumerate(testloader, 0):# Total 20, batch size is 8, so i from 0 to 19
        
        # print('Acutal i = ', i)
        Num_Itera=Num_Itera+1 # Count current iteration of used batches
        
        print('Num of iteration = ', Num_Itera)
        
        F_scores = []

        inputs, labels = data
        labels = labels*255
        # Both of them are torch.Tensor
        # inputs_size: [batch size, # of input channels(image value of each pixel), # of rows, # of columns]
        # labels.size()) [batchsize,1,96,104]
                
        # print('type of labels is ', type(labels))
        # print('type of inputs is ', type(inputs))
        
        outputs = net(inputs)
        _, pred = torch.max(outputs, dim=1)
        
         # Outputs_size: [batch size, # of classes(Probility of being each class), # of rows, # of columns]!
        # pred_size: [batchsize,96,104], Pick the highest probility of belonging to a class at each pixel
        
        #print('Type of inputs = ', type(inputs), ', Shape of inputs = ', inputs.shape)
        #print('Orig output_size = ', outputs.size(), ', Type of outputs = ', type(outputs))
        #print('Orig pred size = ', pred.size(), ', Type of pred is ', type(pred))
        #print('Orig labels size = ', labels.size(), ', Type of labels = ', type(labels))
        # All belong to class 'torch.Tensor'
        
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

        '''
        for j in range(10):
            Flg = 0
            for l in range(104):
                #if max(pred[j,:,l])==1|max(pred[j,:,104-l-1]:
                for m in range(96):
                    if Flg == 2:
                        break
                    else:
                        if pred[j,m,l]==1 or pred[j,m,104-l-1]==1:
                            if pred[j,m,l]==1:
                                Dia_X1 = l
                                Dia_Y1 = m
                                Flg = Flg + 1
                            elif pred[j,m,104-l-1]==1:
                                Dia_X2 = 104-l-1
                                Dia_Y2 = m
                                Flg = Flg + 1
                if Flg == 2:
                    break
            Dia.append(np.sqrt((Dia_X1-Dia_X2)^2+(Dia_Y1-Dia_Y2)^2)-6)
        Ave_Diam=np.mean(Dia)
        print('Diameter is:', Ave_Diam)
        '''
        #Pred=pred.numpy()
        # print('pred row = ', Pred.shape[0], ', pred_column = ', Pred.shape[1])
        # print('labels size = ', labels.size())
        
        output_target.append(pred) 
        # pred store to output_target
        # pred_size is [8(batch size),96,104]
        # labels_size is [8,1,96,104]
        print('output_target_size = ', len(output_target))
        # for i in rang(2)
        # print(' '.join('%5s' % classes[pred[j,]] for j in range(2)))            
        acc = compute_acc(pred, labels.long())
        pred_acc += acc
        total += 96*104*5
        print('Average acc:', pred_acc/total)
        
        Pred_True, Real_True, Corr_True = compute_F(pred, labels.long())
        # Total evaluated cells are 96*104*batch size
        # print('Pred_True = ', len(Pred_True))
        
        for j in range(2): # Based on the number of classes
            F_scores.append(2*Corr_True[j]/(Pred_True[j]+Real_True[j]))
        print('Pred_outside=', Pred_True[0], ', Pred_onside=', Pred_True[1])
        print('Real_outside=', Real_True[0], ', Real_onside=', Real_True[1])
        print('Corre_outside=', Corr_True[0], ', Corre_onside=', Corr_True[1])
        
        print('  F_scores:', F_scores)
        
        # print('input_size = ', input.size())
        # print('orig output_size = ', outputs.size())
        # print('orig labels_size = ', labels.size())
        
        n,c,h,w=outputs.size()# outputs: [batchsize, 2, 96, 104]
        outputs = outputs.transpose(1,2).transpose(2,3).contiguous().view(-1,c) 
        # After this, outputs: [99840,2]: 5*96*104. 5 is batchsize
        LABELS = labels
        labels = labels.view(-1).long()
        # Original labels.size(): [5,1,96,104]
        # Then:[99840]: 5*96*104
        
        print('Latest output_size = ', outputs.size())
        print('After labels_size = ', labels.size())
        
        loss = criterion(outputs, labels)
        # labels: [79872]: (5*1*96*104)
        print('Test loss:',loss.item(), end ='\n\n')
    t4 = time.time()
    print('Profiling time of prediction is {:.4f} sec'.format(t4-t3))    
    
print('type of output_target is ',type(output_target),', size of output_target is ', len(output_target))  
# <class 'list', size is 5
output_target = torch.cat(tuple(output_target))

print('  type of output_target is ',type(output_target)) # class 'torch.Tensor' 
print('Test_output', output_target.shape) # size is [total test datasets,96,104]
# output_target is accumulated over all test samples