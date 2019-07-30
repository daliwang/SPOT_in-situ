# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:48:04 2019

@author: Jian Zhou
"""

import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms

from PIL import Image
import os
import os.path
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def Read_dataset(datadir):
    images = []
    IMAGES = []
    CountImg = 0
    for filename in os.listdir(datadir):# returns a list containing the names of the entries in the directory given by path.
        
        #print('No of images is ', filename[4:-4], 'Type of image NO is ', type(filename[4:-4]))
        IMAGES.append(filename)
        # Since list cannnot be assigned like lst[i]=something unless the list is already initialized.
        # IMAGES contains CountImg number of items. While index starts from 0 to CountImg-1
        CountImg += 1     
    # IMAGES stores from its index 0 to CountImg-1
    #print('Total number of file is ', CountImg)
    
    for filename in os.listdir(datadir):
        IMAGES[int(filename[4:-4])-1]=filename # IMAGES[0] stores 'dat_1.jpg'. 4:-4 means 1 here.
    
    for i in range(CountImg): # 0 ~ CountImg-1
        _image = os.path.join(datadir, IMAGES[i])# The function joins datadir with filename. It returns a path with backslash (\) in windows 
        #print(_image) # : Data\Inputs\Image\dat_X.jpg. datadir is Data\\Inputs\\Image
            
        # _label = os.path.join(labeldir, 'GrdT%s.png' %(filename[3:-4]))
        
     #   print('Type of filname: ', type(IMAGES[i]), ', filname is ', IMAGES[i])
        # print('Type of _image:', type(_image)) # _image is a path for a image
        #print('Shape of _image is ', np.shape(_image))# .shape is ()
        # plt.imshow(_image[111,:,:])
                   
        #############Example of rename directory and print result
        #import os
        #oldpath = os.path.join("C:\\","temp","python")
        #newpath = os.path.join("C:\\","temp","python3")
        #if os.path.exists(oldpath):
        #    os.rename(oldpath, newpath)
        #    print("'{0}' is renamed to '{1}'".format(oldpath,newpath))
        # Output: 'C:\temp\python' is renamed to 'C:\temp\python3'          
        images.append(_image)

    return images


class ModifyDataloder(data.Dataset):
    def __init__(self, datadir, train=True, transform=None):
        self.transform = transform
        self.images = Read_dataset(datadir)

    def __getitem__(self, index):# for i, data in enumerate(inputloader, 0):
        _img = Image.open(self.images[index])        
        #IMG=np.asarray(_img)
        #print('Max of IMG = ', np.max(IMG[:,:]), ', Min of IMG = ', np.min(IMG[:,:]), '\n')# Max:255 and Min:0            
        if self.transform is not None:
            _img = self.transform(_img)

        return _img

    def __len__(self):
        return len(self.images)


class Dataloder():
    def __init__(self):
        normalize = transforms.Normalize(mean=[0.1294],
                                         std=[0.1258])
    # Local-wise normalization 
#96*104 # For E040_Min_13: When # is 402 (class: 0,1); Mean (0.2907)0.2531, Std (0.0199)0.1615. Selected total 81 frames. Mean is 0.2233, Std is 0.0105
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
        #          Max_14: # is 402: Mean is 0.1592, Std is 0.0111. # 107: Mean is 0.1194, Std is 0.0107
                 # Max_15: When # 402, Mean is 0.1698, Std is 0.0105. When # is 174, Mean is 0.1604, Std is 0.0103 

#96*104 # For E040_Min_13: Mean is 0.2338, Std is 0.1581 (238 continuously selected frames for each video) 
        #          Min_14: Mean is 0.284, Std is 0.168 
        #          Min_15: Mean is 0.2504, Std is 0.1671 
        # For E041_Low_13: Mean is 0.2389, Std is 0.1779
        #          Low_14: Mean is 0.2909, Std is 0.1904
        #          low_15: Mean is 0.2479, Std is 0.1701
        # For E042_Target_13: Mean is 0.2589, Std is 0.1832
        #          Target_14: Mean is 0.2841, Std is 0.1802
        #          Target_15: Mean is 0.2679, Std is 0.1774
        # For E043_High_13: Mean is 0.1993, Std is 0.1667
        #          High_14: Mean is 0.2147, Std is 0.1621
        #          High_15: Mean is 0.1957, Std is 0.1483
        # For E044_Max_13: Mean is 0.1190, Std is 0.1261
        #          Max_14: Mean is 0.1393, Std is 0.1185
                 # Max_15: Mean is 0.1294, Std is 0.1258
        
#185*105# For E047_Target_13: Mean is 0.1400, Std is 0.1378
        #          Target_14: Mean is 0.1699, Std is 0.1429
        #          Target_15: Mean is 0.1863, Std is 0.1465
        # For E048_High_13: Mean is 0.0648, Std is 0.1028
        #          High_14: Mean is 0.0776, Std is 0.1043
        #          High_15: Mean is 0.0776, Std is 0.1055
        # For E049_Max_13: Mean is 0.0696, Std is 0.1105
        #          Max_14: Mean is 0.0766, Std is 0.1114
                 # Max_15: Mean is 0.0856, Std is 0.1164

#92*104 # For E050_Min_13: Mean is 0.1133, Std is 0.164 (238 continuously selected frames!) (T_0=173)
        #          Min_14: Mean is 0.1312, Std is 0.1797 (T_0=173)
        #          Min_15: Mean is 0.1169, Std is 0.1601(T_0=174) 
        #        2_Min_15: Mean is 0.1170, Std is 0.1600(T_0=175). Different T0 Does not impact CNN prediction !! 
        # For E051_Low_13: Mean is 0.0844, Std is 0.1264
        #          Low_14: Mean is 0.1209, Std is 0.1638
        #          low_15: Mean is 0.1317, Std is 0.1710
        # For E052_Target_13: Mean is 0.1126, Std is 0.1667
        #          Target_14: Mean is 0.1287, Std is 0.169
        #          Target_15: Mean is 0.1325, Std is 0.1666
        # For E053_High_13: Mean is 0.0805, Std is 0.1479
        #          High_14: Mean is 0.0781, Std is 0.1361
        #          High_15: Mean is 0.0862, Std is 0.1399
        # For E054_Max_13: Mean is 0.1185, Std is 0.1847
        #          Max_14: Mean is 0.0667, Std is 0.1572
                 # Max_15: Mean is 0.0722, Std is 0.1652               
                 
#119*103 # For E059_Target_13: Mean is 0.4586, Std is 0.1909
        #          Target_14: Mean is 0.375, Std is 0.1679
        #          Target_15: Mean is 0.3805, Std is 0.1672
        # For E060_High_13: Mean is 0.4844, Std is 0.2137
        #          High_14: Mean is 0.4527, Std is 0.2073
        #          High_15: Mean is 0.4558, Std is 0.2109
        # For E061_Max_13: Mean is 0.477, Std is 0.23
        #          Max_14: Mean is 0.4546, Std is 0.2351
                 # Max_15: Mean is 0.4188, Std is 0.2418                 

#70*79 (Max, High, Target) && 60*79 (Low, Min)
        # For E063_Min_13: Mean is 0.2136, Std is 0.1299 (238 continuously selected frames!)
        #          Min_14: Mean is 0.1802, Std is 0.1201
        #           Min_15: Mean is 0.1959, Std is 0.1332 
        # For E064_Low_13: Mean is 0.1532, Std is 0.1142
        #          Low_14: Mean is 0.1785, Std is 0.1230
        #           low_15: Mean is 0.1962, Std is 0.1396
        # For E065_Target_13: Mean is 0.1708, Std is 0.1418
        #          Target_14: Mean is 0.1742, Std is 0.1375
        #           Target_15: Mean is 0.1005, Std is 0.0719
        # For E066_High_13: Mean is 0.1975, Std is 0.1447
        #          High_14: Mean is 0.1587, Std is 0.1466
        #           High_15: Mean is 0.1312, Std is 0.1207
        # For E067_Max_13: Mean is 0.2994, Std is 0.1953
        #          Max_14: Mean is 0.1985, Std is 0.1835
                 #  Max_15: Mean is 0.1398, Std is 0.1213
        # Average mean and std based on above data are 0.1792, std is 0.1349
                 
#74*95(Max); 84*112(Others)
        # For E068_Min_13: Mean is 0.3674, Std is 0.0882 (238 continuously selected frames!) (T_0=173)
        #          Min_14: Mean is 0.2112, Std is 0.0576 (T_0=173)
        #          Min_15: Mean is 0.3828, Std is 0.0841(T_0=174) 
        # For E069_Low_13: Mean is 0.4573, Std is 0.099
        #          Low_14: Mean is 0.3869, Std is 0.0713
        #          low_15: Mean is 0.616, Std is 0.1062. Mean is the biggest!
        # For E070_Target_13: Mean is 0.6609, Std is 0.1206
        #          Target_14: Mean is 0.5516, Std is 0.0878
        #          Target_15: Mean is 0.3603, Std is 0.0649
        # For E071_High_13: Mean is 0.3878, Std is 0.0845
        #          High_14: Mean is 0.5983, Std is 0.1214
        #          High_15: Mean is 0.5877, Std is 0.1219
        # For E072_Max_13: Mean is 0.4281, Std is 0.1216
        #          Max_14: Mean is 0.3224, Std is 0.1126
                 # Max_15: Mean is 0.2945, Std is 0.1108
        # Average mean and std based on above data are 0.4409, std is 0.0968. 
        # Only this material combination uses common mean and std!!               

#100*81(Low & Target); 90*66(High & Max)
        # For E075_Low_13: Mean is 0.3206, Std is 0.1537
        #          Low_14: Mean is 0.3366, Std is 0.1490
        #          low_15: Mean is 0.2806, Std is 0.1245
        # For E076_Target_13: Mean is 0.4882, Std is 0.2235
        #          Target_14: Mean is 0.4731, Std is 0.2174
        #          Target_15: Mean is 0.4297, Std is 0.2023
        # For E077_High_13: Mean is 0.3667, Std is 0.1612
        #          High_14: Mean is 0.4378, Std is 0.2038
        #          High_15: Mean is 0.4746, Std is 0.2295
        # For E078_Max_13: Mean is 0.2997, Std is 0.1131
        #          Max_14: Mean is 0.4294, Std is 0.1812
                 # Max_15: Mean is 0.3269, Std is 0.1748
         
    # Pixel-wise normalization regarding E040-E044      
#96*104 # For E040_Min_13: When # is Selected total 238 frames. Mean is 0.574, Std is 0.3219
        #          Min_14: When # is Selected total 238 frames. Mean is 0.5758, Std is 0.3223
        #          Min_15: When # is Selected total 238 frames. Mean is 0.5763, Std is 0.3234
        
        # For E041_Low_13: When # is Selected total 238 frames. Mean is 0.5715, Std is 0.3272
        #          Low_14: When # is Selected total 238 frames. Mean is 0.5754, Std is 0.3282
        #          low_15: When # is Selected total 238 frames. Mean is 0.5788, Std is 0.3298
        
        # For E042_Target_13: When # is Selected total 238 frames. Mean is 0.5797, Std is 0.3297
        #          Target_14: When # is Selected total 238 frames. Mean is 0.581, Std is 0.331
        #          Target_15: When # is Selected total 238 frames. Mean is 0.5781, Std is 0.3315
        
        # For E043_High_13: When # is Selected total 238 frames. Mean is 0.5743, Std is 0.3346
        #          High_14: When # is Selected total 238 frames. Mean is 0.5725, Std is 0.3361
        #          High_15: When # is Selected total 238 frames. Mean is 0.5743, Std is 0.3348
        
        # For E044_Max_13: # is Selected total 238 frames. Mean is 0.5554, Std is 0.3371
        #          Max_14: # is Selected total 238 frames. Mean is 0.5041, Std is 0.3388
                 # Max_15: When Selected total 238 frames. Mean is 0.5408, Std is 0.3284       

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        #inputset = ModifyDataloder('Data\\Inputs\\Image', train=False, transform=transform_test) # What's the train=False is used for??!!
        inputset = ModifyDataloder('Data\\Images\\Min\\E050\\000015\\Selected', train=False, transform=transform_test)
        # Read input images datasets and normalize
        
        kwargs = {'num_workers': 0, 'pin_memory': True}

        inputloader = torch.utils.data.DataLoader(inputset, batch_size=
            1, shuffle=False, **kwargs)
        # batch_size changes

        self.inputloader = inputloader
        self.Num_input = inputset.__len__()
    
    def getloader(self):
        return self.inputloader, self.Num_input

#if __name__ == "__main__":
#  trainset = ModifyDataloder('Data\\train\\Image', 'Data\\train\\Label', train=True)
#    testset = ModifDataloder('Data\Image', 'Data\Label', train=False)
#    print(trainset.classes)
#    print(len(testset))