#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:38:50 2021

@author: grondat1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNN(nn.Module):
    
    def __init__(self,
                 input_size,
                 embedding_dim,
                 kernel_sizes=(3, 4, 5),
                 kernel_num=10,
                 dropout=0.1,
                 class_num=2):
        
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim) #embedding layer
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (size, embedding_dim)) for size in kernel_sizes]) #list of convolution layers with different kernel sizes
        self.dropout = nn.Dropout(dropout) #dropout layer
        self.fc = nn.Linear(kernel_num*len(kernel_sizes), class_num) #fully connected layer
    
    def forward(self, x): # Shape of x: (batch_size, sequence_length)
        
        emb = self.embedding(x) # embded word sequence: (batch_size, sequence_length, embedding_dim)
        emb = emb.unsqueeze(1) # add dimension: (batch_size, 1, sequence_length, embedding_dim)
        y_conv = [conv(emb) for conv in self.convs] # convolution for all kernel sizes: (batch_size, kernel_num, kernel_appl_num, 1) "kernel_appl_num" = how many times has the kernel fit the data based on size and stride
        y_relu = [F.relu(c).squeeze(3) for c in y_conv] # ReLu over conv-results and remove last dim: (batch_size, kernel_num, kernel_appl_num)
        y_mp = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in y_relu] # max pool relu results: (batch_size, kernel_num)
        y_cat = torch.cat(y_mp, 1) # concatenate max-pooled results for all kernel sizes: (batch_size, kernel_num*len(kernel_sizes))
        y_do = self.dropout(y_cat) # dropout: (batch_size, kernel_num*len(kernel_sizes))
        o = self.fc(y_do) # apply linear layer: (batch_size, class_num)
        return o