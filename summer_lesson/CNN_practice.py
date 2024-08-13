import os 
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt


## first part : download data (very famous dataset called Mnist)
"""
your code
"""

## second part : Batch according to batch size and preparing testing dataset
"""
your code
"""

## third part : Create NN
"""
basic form:
class CNN(torch.nn.Module): # class a Network and input a torch module
    def __init__(self, n_feature, n_hidden, n_output):
        super(CNN, self).__init__() # To inherit things from Net, the standard process must be added

    def forward(self, x):
        
        return x 

optimizer = torch.optim.SGD()
loss_func = torch.nn.MSELoss()
"""

## third part : training and testing
"""
your code
"""