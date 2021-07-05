#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:40:31 2021

@author: grondat1
"""

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def pad_batch(batch, size, pad_ix=0):
    return [l+[pad_ix for i in range(size-len(l))] for l in batch]

class AttnLSTM(nn.Module):
    
    def __init__(self,
                 vocabulary_size,
                 embedding_dim=300,
                 hidden_dim=300,
                 bidir=False,
                 num_classes=2,
                 num_layers=1,
                 dropout_p=0.1,
                 attn_length=10,
                 device='cpu'):
        
        super(AttnLSTM, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.attn_length = attn_length
        self.bidir = bidir
        
        self.mult = 1
        if bidir:
            self.mult = 2
        
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = nn.Linear(self.hidden_dim, self.attn_length)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // self.mult, num_layers = num_layers, bidirectional=bidir)
        self.clf = nn.Linear(2*self.hidden_dim, num_classes)

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.mult * self.num_layers, batch_size, self.hidden_dim // self.mult)).to(self.device),
                Variable(torch.zeros(self.mult * self.num_layers, batch_size, self.hidden_dim // self.mult)).to(self.device))
    
    def forward(self, input_batch, batch_size, pack=True):
        self.hidden = self.init_hidden(batch_size)
        
        batch_lengths = torch.tensor([len(l) for l in input_batch])
        max_len = max(batch_lengths).item()
        batch_pad = torch.tensor(pad_batch(input_batch, max_len))
        batch_lengths, perm_idx = batch_lengths.sort(0, descending=True)
        input_batch = batch_pad[perm_idx]
        input_embedding = self.embedding(input_batch)
        input_embedding = self.dropout(input_embedding)
        
        if pack:
            input_embedding = pack_padded_sequence(input_embedding, list(batch_lengths), batch_first=True)
        else:
            input_embedding = input_embedding.permute(1,0,2)
        
        lstm_out, (h,c) = self.lstm(input_embedding, self.hidden)
        if pack:
            lstm_out = pad_packed_sequence(lstm_out)[0]
        
        if self.bidir:
            h = torch.cat((h[-2], h[-1]), 1)
            c = torch.cat((c[-2], c[-1]), 1)

        attn_weights = F.softmax(self.attn(h), dim=1)
        
        lstm_out = lstm_out[:self.attn_length]
        if lstm_out.size(0) < self.attn_length:
            rest = torch.zeros(self.attn_length - lstm_out.size(0), batch_size, lstm_out.size(2)).to(self.device)
            lstm_out = torch.cat((lstm_out, rest), 0)
        
        lstm_out = lstm_out.transpose(0,1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), lstm_out).view(1, batch_size, -1)
        
        clf_in = torch.cat((h, attn_applied[0]), 1)
        clf_out = self.clf(clf_in)
        clf_out = F.log_softmax(clf_out, dim=1)
            
        return clf_out