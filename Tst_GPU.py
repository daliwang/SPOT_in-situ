# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:27:27 2019

@author: Jian Zhou
"""
import torch
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"]="0"

x = torch.randn(1)
# print(x)
# print(x.item())

# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

'''
Output should be the following:
tensor([1.9641], device='cuda:0')
tensor([1.9641], dtype=torch.float64)
'''