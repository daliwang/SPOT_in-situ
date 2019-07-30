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
# print(net) # claim the network model structure
dataloader = Dataloder()
classes, trainloader, testloader = dataloader.getloader()

print('trainloader_batch_size = ', trainloader.batch_size)
print('num of trainloader = ', trainloader.__len__(),'\n')

criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.01,1.0]))# Initial: 0.005,1.0
#combines nn.LogSoftmax() and nn.NLLLoss(). The popular loss function
# A loss function takes the (output, target) pair of inputs, and computes a value that estimates how 
# far away the output is from the target.

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
#Stochastic Gradient Descent simplest update weights   

def adjust_lr(optimizer, epoch, lr_decay=10):
    lr = 0.01 *(0.1**((epoch-1)//lr_decay))
    if (epoch-1) % lr_decay == 0:
        print('LR is set to: ', lr)
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # Decreasing learning rate per 10 epoch to try to increase accuracy. I refer to the examples online
    # weight = weight - learning_rate * gradient
    
def compute_acc(pred, label): # calculate accuracy
    pred = pred.unsqueeze(1)
    # Increase a dimension of pred in order to compare with label
    # print((label == 2).sum())
    #print('pred size:', pred.shape)
    #print('label size:', label.shape)
    # print(' '.join('%5s' % classes[pred[j]] for j in range(2)))
    
    acc_sum = (pred == label).sum()
    # Totally 96*104*batchsize. predictions/comparisons.
    return acc_sum.item()
    
def compute_F(pred, label): # calculate F1 scores for 3 labels
    pred = pred.unsqueeze(1).numpy()
    # pred_size: [batchsize,1,96,104]
    label = label.numpy()
    # labels_size is [batchsize,1,96,104]
    # print(Counter(label.reshape(-1)))
    # Totally 96*104*2 predictions/comparisons cause batch size is 2
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
        # classes = ['outside','onside','inside'] defined in find_classes():
    
    return Pred_Tru, Real_Tru, Corr_Tru

Num_Itera= 0

net.load_state_dict(torch.load('model.ckpt'))  

########### Training datasets follow an order: [1 2 5 9 8 4 10 3 6 7], from Min_13 to Max_14(only NO.13 & NO. 14 videos are adopted), total 10 datasets
########### Each datasets are formed from strictly selected frames (after T_0) based on ground truth area!!!!

for epoch in range(5):  # loop over the dataset multiple times
    # Which parameters are updated continuslly over each epoch ??!!
    
    t1 = time.time()
    # t[L]=time.time()
    pred_acc, total = 0, 0
    running_loss = 0.0
    
    print('Train session epoch: {:} '.format(epoch+1))
    adjust_lr(optimizer, epoch, lr_decay=10)
    
    for i, data in enumerate(trainloader, 0):
        
        Num_Itera=Num_Itera+1 # Count current number of batches
        
        print('  Num of iteration = ', Num_Itera)
        # get the inputs
        inputs, labels = data
        labels = labels*255
        #print('labels_size = ', labels.size()) # [batchsize,1,96,104]
        # each data has 2 images(batch size is 2), in total 96*104*2 cells
        
        # inputs_size: [batch size, # of input channels(image value of each pixel), # of rows, # of columns]
        # [batchsize,1,96,104]
        # print('inputs_size = ', inputs.size())
        # print('Max of inputs is ', np.max(inputs.numpy()), 'Min is ', np.min(inputs.numpy())) #<class 'numpy.ndarray'>

        optimizer.zero_grad()
       # Zero the gradient buffers of all parameters & zero the parameter gradients
       # clear the existing gradients, else gradients will be accumulated to existing gradients.
       
        # forward + backward + optimize        
        outputs = net(inputs)
        _, pred = torch.max(outputs, dim=1)
        # Outputs_size: [batch size, # of classes, # of rows, # of columns]! e.g.,[batchsize, 2, 96, 104]
        
        # pred_size: [batchsize,96,104], Pick the highest probility of belonging to a class at each pixel       
        #print('pred_size = ', pred.size()) # [bath size, 96, 104]   
        # print('pred_acc',pred_acc)
        # print('total',total)
        
        acc = compute_acc(pred, labels.long())
        pred_acc += acc
        total += 96*104*5 # Identify batch size!
        print('Average acc:', pred_acc/total)
        # batch size is 2(images), so all have 96*104*2 cells(predicted labels)
        
        # print('Orig labels_size = ', labels.size())
        # print('Orig output_size = ', outputs.size())
        
        n,c,h,w=outputs.size()
        outputs = outputs.transpose(1,2).transpose(2,3).contiguous().view(-1,c)
        # Before this, outputs: [batchsize, 2, 96, 104]
        # After, outputs: [79872,3] !

        LABELS = labels
        labels = labels.view(-1).long() # Make labels to be 1 diamention by multiplying all original diamentions 
        # make it the same shape
        # labels: [99840]: (batchsize*1*96*104)
        print(Counter(labels.numpy().reshape(-1)))
        
        # print('Latest output_size = ', outputs.size())
        # print('After labels_size = ', labels.size())
        
        loss = criterion(outputs, labels)# outputs: [batchsize, 2, 96, 104]. labels [batchsize,1,96,104]
        # outputs: 2 probibilities of being each class at each pixel. labels: actual class of each pixel
              
        loss.backward() # backpropagate the error (gradients back into the network’s parameters)
        optimizer.step() # Does the update to get new weights!

        # print statistics
        running_loss += loss.item() 
    t2 = time.time()
    print('Profiling time is {:.4f} sec'.format(t2-t1), end ='\n\n')
    # print(end ='\n\n')
    
    torch.save(net.state_dict(), 'model.ckpt') 