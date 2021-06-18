#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 13:34:04 2021

@author: grondat1
"""

import os
import json
import argparse
import itertools
import collections
import numpy as np

# simple formal language dataset: (subset of) all combinations with optional restrictions on tokens for positions
class Dataset():
    
    def __init__(self,
                 vocabulary=['a', 'b'], # tokens to concatenate to sequences
                 min_len=1, # smallest sequence
                 max_len=0, # longest sequence; len(vocabulary) if 0
                 voc_in_position={}, # dict from positions to tokens only allowed in those positions
                 discard_from_position={}, # dict from positions to list of tokens in vocabulary: don't put these tokens in these positions
                 max_size=10000): # maximum number of datapoints in resulting dataset)

        sequences = []
        max_len = len(vocabulary) if not max_len else max_len
        
        # make vocabulary and restrictions available for set operations
        vocabulary = set(vocabulary)
        voc_in_position = {p:set(voc_in_position[p]) for p in voc_in_position}
        discard_from_position = collections.defaultdict(set, {p:set(discard_from_position[p]) for p in discard_from_position})
        
        # how many combinations would be available
        max_comb = 0
        for n in range(min_len, max_len+1):
            max_comb += len(vocabulary)**n
            if max_comb > max_size:
                break
        
        # if all possible combinations don't exceed max_size, do them all
        if max_comb <= max_size:
            
            for n in range(min_len, max_len+1): # do sequences of each size between min_len and max_len
                n_seq = []
                
                for i in range(n): # all tokens for each position in the sequence
                    pos_voc = voc_in_position[i] if i in voc_in_position else vocabulary
                    pos_voc -= discard_from_position[i] # don't use tokens discarded in the current position
                    n_seq.append(list(pos_voc))
                
                n_seq = list(itertools.product(*n_seq))
                sequences += [''.join(seq) for seq in n_seq]
                
        # if all possible combinations exceed max_size, do random selection of size max_size
        else:        
            sequences = set() # use set for avoiding repetition of randomly generated datapoints
            lengths = list(range(min_len, max_len+1)) # possible sequence lengths
            
            while len(sequences) < max_size: # add new sequence until max_size is reached
                n = np.random.choice(lengths) # choose random sequence length
                n_seq_rand = []
                
                for i in range(n): # random token for each position in the sequence
                    pos_voc = voc_in_position[i] if i in voc_in_position else vocabulary
                    pos_voc -= discard_from_position[i]
                    rand_token = np.random.choice((list(pos_voc)))
                    n_seq_rand.append(rand_token)
                    
                sequences.add(''.join(n_seq_rand))
                
            sequences = list(sequences)
            
        self.src = sorted(sequences)
        self.vocabulary = vocabulary
        self.min_len = min_len
        self.max_len = max_len
        self.voc_in_position = voc_in_position
        self.discarf_from_position = discard_from_position
        self.max_size = max_size
    
    # set target values
    def set_tgt(self,
                task='copy', # copy/different/reverse/replace/uppercase/count/custom
                custom_label='X'): # label in custom task
        
        assert task in ['copy', 'different', 'reverse', 'replace', 'uppercase', 'count', 'custom']
        
        # sequences mapped to themselves
        if task=='copy':
            self.tgt = self.src
        
        # sequences mapped to different sequences with edit distance between 1 and max_edit_distance
        elif task == 'different':
            if len(self.src) < 2:
                print('At least 2 source expressions needed to make targets')
            else:
                self.tgt = []
                for seq in self.src:
                    tgt_seq = np.random.choice(self.src)
                    while seq == tgt_seq:
                        tgt_seq = np.random.choice(self.src)
                    self.tgt.append(tgt_seq)
            assert [s!=t for (s,t) in zip(self.src, self.tgt)] # just a sanity check
        
        # sequences mapped to their reversal
        elif task=='reverse':
            self.tgt = [seq[::-1] for seq in self.src]
                
        # sequences mapped to uppercase variants
        elif task=='uppercase':
            self.tgt = [seq.upper() for seq in self.src]
        
        # sequqneces mapped to token counts (for optionally specified tokens; all tokens counted by default)
        elif task=='count':
            self.tgt = [len(seq) for seq in self.src]
        
        # sequences mapped to user-given custom label
        elif task=='custom':
            self.tgt = [custom_label for seq in self.src]
        
        assert len(self.src) == len(self.tgt) # just a sanity check
            
# specify dataset params and labels, create dataset, save to file
def main(args):
    
    # set random seed if specified
    if args.random_seed:
        np.random.seed(args.random_seed)
    
    # make dataset
    ds = Dataset(vocabulary=args.vocabulary,
                 min_len=args.min_len,
                 max_len=args.max_len,
                 voc_in_position=args.voc_in_position,
                 discard_from_position=args.discard_from_position,
                 max_size=args.max_size)
    
    # make targets
    ds.set_tgt(task=args.task)
    
    # make src-tgt pairs
    src_tgt = list(zip(ds.src, ds.tgt))
    
    if args.shuffle:
        np.random.shuffle(src_tgt)
    
    # save folder name = task + vocabulary[:5] + differentiating int
    cont_str = '...' if len(ds.vocabulary) > 5 else ''
    save_folder = os.path.join(args.save_folder, '{0}_{1}'.format(args.task, ''.join(sorted(ds.vocabulary)[:5]) + cont_str))
    
    # add int to save_folder name for differentiating between variants
    folder_int = 1
    while os.path.exists('{0}_{1}'.format(args.save_folder, folder_int)):
        folder_int += 1
    save_folder = '{0}_{1}'.format(save_folder, folder_int)
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # save dataset parameters to json file
    params = {'task':args.task,
              'vocabulary':args.vocabulary,
              'min_len':args.min_len,
              'max_len':args.max_len,
              'voc_in_position':args.voc_in_position,
              'discard_from_position':args.discard_from_position,
              'shuffle':args.shuffle,
              'random_seed':args.random_seed,
              'max_size':args.max_size}
    
    with open(os.path.join(save_folder, args.params_fname), 'w') as f:
        json.dump(params, f)
    
    # save data to a txt-file: tokens separated by separator, src and tgt separated by delimiter
    with open(os.path.join(save_folder, args.data_fname), 'w') as f:
        for src, tgt in src_tgt:
            f.write(args.separator.join(src) + args.delimiter + args.separator.join(tgt) + '\n') # add separators and delimiters


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-task', '--task', required=True) # copy/reverse/replace/uppercase/count/custom
    arg_parser.add_argument('-voc', '--vocabulary', type=list, required=True)
    arg_parser.add_argument('--min_len', type=int, default=1)
    arg_parser.add_argument('--max_len', type=int, default=0) # vocabulary length by default
    arg_parser.add_argument('--voc_in_position', type=dict, default={})
    arg_parser.add_argument('--discard_from_position', type=dict, default={})
    arg_parser.add_argument('--custom_label', default='X')
    arg_parser.add_argument('--max_size', type=int, default=10000)
    arg_parser.add_argument('--shuffle', type=bool, default=True) # randomize order of src-tgt pairs
    arg_parser.add_argument('--random_seed', type=int, default=12345) # set random seed
    arg_parser.add_argument('--separator', default=' ') # separates tokens in saved txt-file
    arg_parser.add_argument('--delimiter', default='\t') # separates src from tgt in saved txt-file
    arg_parser.add_argument('--save_folder', default='data/')
    arg_parser.add_argument('--params_fname', default='params.json')
    arg_parser.add_argument('--data_fname', default='src_tgt.txt')
    args = arg_parser.parse_args()
    main(args)