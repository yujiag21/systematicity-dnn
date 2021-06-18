#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 20:42:44 2021

@author: grondat1
"""

# Training language model (LM) for sequence (or sequence pair) classification tasks
# Takes txt-files with src-tgt pairs as arguments, pools together and trains LM
# Argument files created with create_datasets.py

import os
import logging
import argparse
import numpy as np
from simpletransformers.language_modeling import (LanguageModelingModel, LanguageModelingArgs)

def main(args):
    
    # Make training data file from original source files containing src-tgt pairs
    
    lm_train_data = []
    
    for fpath in args.data:
        data = open(fpath, 'r').readlines()
        data = [l.strip() for l in data]
        data = [l.split(args.delimiter) for l in data]
        if args.only_src:
            lm_train_data += [s for (s,t) in data]
        else:
            lm_train_data += [s for (s,t) in data] + [t for (s,t) in data]
    
    lm_train_data = sorted(set(lm_train_data))
    np.random.shuffle(lm_train_data)
    
    # get vocabulary from training data (discard whitespace)
    lm_train_voc = sorted(set(''.join(lm_train_data)) - {' '})
    
    if not os.path.exists(args.save_data_folder):
        os.makedirs(args.save_data_folder)
    
    # filepath to save training data: named by vocabulary[:5]
    cont_str = '...' if len(lm_train_voc) > 5 else ''
    train_file = ''.join(lm_train_voc[:5]) + cont_str
    train_file = os.path.join(args.save_data_folder, train_file)
    
    # add int to save_folder name for differentiating between variants
    folder_int = 1
    while os.path.exists('{0}_{1}.txt'.format(train_file, folder_int)):
        folder_int += 1
    save_train_fname = '{0}_{1}'.format(train_file, folder_int) + '.txt'
    
    with open(train_file, 'w') as f:
        for seq in lm_train_data:
            f.write(seq.strip() + '\n')
    
    # Train LM
    
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    
    model_args = LanguageModelingArgs()
    model_args.reprocess_input_data = True
    model_args.overwrite_output_dir = True
    model_args.num_train_epochs = args.epochs
    model_args.dataset_type = "simple"
    model_args.vocab_size = args.vocab_size
    
    model = LanguageModelingModel(args.model, None, args=model_args, train_files=train_file, use_cuda=args.use_cuda)
    model.train_model(train_file, eval_file=train_file)
    result = model.eval_model(train_file)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--data', nargs='*') # paths to txt-files with data
    arg_parser.add_argument('--delimiter', default='\t') # separates src from tgt in data file
    arg_parser.add_argument('--only_src', action='store_true') # use only src from data file for training
    arg_parser.add_argument('--save_data_folder', default='lm_training_data/') # separates src from tgt in data file
    arg_parser.add_argument('-m', '--model', default='bert') # model type
    arg_parser.add_argument('--epochs', type=int, default=5) # number of training epochs
    arg_parser.add_argument('--vocab_size', type=int, default=30000) # max vocabulary size
    arg_parser.add_argument('--random_seed', type=int, default=12345) # fix random seed
    arg_parser.add_argument('--use_cuda', action='store_true') # gpu/cpu
    args = arg_parser.parse_args()
    
    if args.random_seed:
        np.random.seed(args.random_seed)
    
    main(args)