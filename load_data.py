import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms

from PIL import Image
import os
import os.path
import numpy as np
from collections import Counter

def find_classes():
    classes = ['outside','onside']
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    # class[0]: outside, class[1]: onside
    # print('classes', class_to_idx)
    
    return classes, class_to_idx


def Read_dataset(datadir, labeldir, class_to_idx):
    images = []
    labels = []
    for filename in os.listdir(datadir):
            _image = os.path.join(datadir, filename)
            _label = os.path.join(labeldir, 'GrdT%s.png' %(filename[3:-4]))
            # Name format of groundtruth figure
            
            images.append(_image)
            labels.append(_label)
            
    # print('shape of labels is ', np.shape(labels)) # list, shape is number of input data
    
    return images, labels


class ModifyDataloder(data.Dataset):
    def __init__(self, datadir, labeldir, train=True, transform=None):
        classes, class_to_idx = find_classes()
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform

        self.images, self.labels = Read_dataset(datadir, labeldir,
            class_to_idx)
        
        assert (len(self.images) == len(self.labels))

    def __getitem__(self, index):
        
        #print('type of self.labels is ', type(self.labels),', shape of self.labels is ', np.shape(self.labels), '\n') # class 'str'
        # self.labels type is 'list', shape is (50,). self.labels[index] type is str.
        _img = Image.open(self.images[index])
        _label = Image.open(self.labels[index])
        
        # print('type of _label is ', type(_label), ', shape of _label is ', np.shape(_label), '\n') # class 'PIL.PngImagePlugin.PngImageFile'. Shape is [96,104]
        
        if self.transform is not None:
            _img = self.transform(_img)
        _label = transforms.ToTensor()(_label) 
        #print('  Type of _label is ', type(_label), ', Shape of _label is ', _label.shape)# Type is torch.tensor. Shape is [1,96,104]

        return _img, _label

    def __len__(self):
        return len(self.images)


class Dataloder():
    def __init__(self):
        normalize = transforms.Normalize(mean=[0.4331],
                                         std=[0.0039]) # 
        # Mean and std of datasets:
        # For E044_Max_13: When # is 401 (class: 0,1); Mean 0.1689, Std 0.0206. Frames after T_0(11~400 according to MATLAB)):mean0.1712,std0.0207 
                 # Max_14: When # 402, Mean is 0.1592, Std is 0.0111. When # is 392, Mean is 0.1604, Std is 0.0109. 
                 # Max_15: When # 402, Mean is 0.1698, Std is 0.0105. When # is 392, Mean is 0.1712, Std is 0.0106.         
        # For E040_Min_13: When # is 402 (class: 0,1); Mean 0.2531, Std 0.0199. Selected total 161 frames. Mean is 0.1865, Std is 0.0167
        #          Min_14: When # is 401 (class: 0,1); Mean 0.3265, Std 0.0223. Selected total 283 frames. Mean is 0.3369, Std is 0.0241
        #          Min_15: When # is 402 (class: 0,1); Mean 0.3168, Std 0.0258. Selected total 312 frames. Mean is 0.3086, Std is 0.0259      
        
        ##################
        # For E040_Min_13: When # is 402 (class: 0,1); Mean 0.2907(2531), Std 0.0199. Selected total 81 frames. Mean is 0.2233, Std is 0.0105
        #          Min_14: When # is 401 (class: 0,1); Mean 0.3265, Std 0.0223. Selected total 107 frames. Mean is 0.4429, Std is 0.0013
        #          Min_15: When # is 402 (class: 0,1); Mean 0.3168, Std 0.0258. Selected total 68 frames. Mean is 0.4331, Std is 0.0039
        
        # For E041_Low_13: When # is 402 (class: 0,1); Mean 0.3330, Std 0.0189. Selected total 128 frames. Mean is 0.3164, Std is 0.0196
        #          Low_14: When # is 402 (class: 0,1); Mean 0.3647, Std 0.0272. Selected total 34 frames. Mean is 0.1831, Std is 0.0063
        #          low_15: When # is 401 (class: 0,1); Mean 0.3257, Std 0.0257. Selected total 287 frames. Mean is 0.2962, Std is 0.0255
        
        # For E042_Target_13: When # is 402 (class: 0,1); Mean 0.3237, Std 0.0236. Selected total 88 frames. Mean is 0.2905, Std is 0.0333
        #          Target_14: When # is 401 (class: 0,1); Mean 0.3647, Std 0.0258. Selected total 209 frames. Mean is 0.3871, Std is 0.0241
        #          Target_15: When # is 401 (class: 0,1); Mean 0.3333, Std 0.0283. Selected total 230 frames. Mean is 0.3794, Std is 0.0161
        
        # For E043_High_13: When # is 402 (class: 0,1); Mean 0.2814, Std 0.0243. Selected total 232 frames. Mean is 0.3623, Std is 0.0125
        #          High_14: When # is 401 (class: 0,1); Mean 0.2853, Std 0.0246. Selected total 211 frames. Mean is 0.3356, Std is 0.0199
        #          High_15: When # is 401 (class: 0,1); Mean 0.2977, Std 0.0194. Selected total 163 frames. Mean is 0.2771, Std is 0.0224
        
        # For E044_Max_13: # is 401 (class: 0,1); Mean 0.1689, Std 0.0206. Selected total 345 frames: mean is 0.1752,std is 0.0152
        #          Max_14: # is 402: Mean is 0.1592, Std is 0.0111. Selected total 107: Mean is 0.1194, Std is 0.0107
                 # Max_15: # is 402, Mean is 0.1698, Std is 0.0105. Selected total is 174: Mean is 0.1604, Std is 0.0103 
        

        # data2_38: 0.161,0.0175; data1_38: 0.2677,0.0196
        # data2_401:0.1689,0.0206; data1_402: 0.2531, 0.0199
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        trainset = ModifyDataloder('Data\\train\\Image', 'Data\\train\\Label', train=True, transform=transform_train)
        testset = ModifyDataloder('Data\\test\\Image', 'Data\\test\\Label', train=False, transform=transform_test)
        # Read datasets and normalize
        # print('num of trainset = ', trainset.__len__())
        
        # Num_input= trainset.__len__()
        
        kwargs = {'num_workers': 0, 'pin_memory': True}
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=
            5, shuffle=True, **kwargs) # Identify batch size
        testloader = torch.utils.data.DataLoader(testset, batch_size=
            5, shuffle=False, **kwargs) # Identify batch size
        # batch_size changes
        # print('num of trainloader = ', trainloader.)
        
        
        self.classes = trainset.classes
        self.trainloader = trainloader 
        self.testloader = testloader
        
        #self.NumInput = trainset.__len__()        
    
    def getloader(self):
        return self.classes, self.trainloader, self.testloader #, self.NumInput



#if __name__ == "__main__":
#  trainset = ModifyDataloder('Data\\train\\Image', 'Data\\train\\Label', train=True)
#    testset = ModifDataloder('Data\Image', 'Data\Label', train=False)
#    print(trainset.classes)
#    print(len(testset))
