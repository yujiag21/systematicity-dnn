# Fine-tuning a LM to sequence (pair) classification

import os
import shutil
import logging
import argparse
import numpy as np
import pandas as pd
from simpletransformers.classification import (ClassificationModel, ClassificationArgs)

def train_clf(args):
    
    # Make training data from original source files containing src-tgt pairs
    
    clf_train_data = []
    
    # for mapping tgt to int as label
    tgt_to_int = {}
    max_int = 0
    
    # for documentation: mapping from label ints to names (data path if pairs, tgt otherwise)
    label_names = {}
    
    for i,fpath in enumerate(args.data):
        data = open(fpath, 'r').readlines()
        data = [l.strip() for l in data]
        data = [l.split(args.delimiter) for l in data]
        
        # classifying pairs: format [src, tgt, label]; each file has a unique label
        if args.pairs:
            clf_train_data += [[s,t,i] for (s,t) in data]
            label_names[i] = fpath
        
        # classifying src to tgt: format [src, tgt]
        else:
            for (s,t) in data:
                if t not in tgt_to_int:
                    tgt_to_int[t] = max_int
                    label_names[max_int] = t
                    max_int += 1
                clf_train_data.append([s, tgt_to_int[t]])
    
    np.random.shuffle(clf_train_data)
    
    train_df = pd.DataFrame(clf_train_data)
    
    if args.pairs:
        train_df.columns = ["text_a", "text_b", "labels"]
    else:
        train_df.columns = ["text", "labels"]
    
    model_args = ClassificationArgs()
    model_args.overwrite_output_dir = True
    model_args.reprocess_input_data = True
    model_args.evaluate_during_training = True
    model_args.dataset_type = "simple"
    model_args.early_stopping_consider_epochs = True
    model_args.early_stopping_patience = 1
    
    model_args.num_train_epochs = args.epochs
    model_args.learning_rate = args.learning_rate
    model_args.vocab_size = args.vocab_size
    model_args.train_batch_size = args.batch_size
    model_args.eval_batch_size = args.batch_size
    model_args.use_early_stopping = args.use_early_stopping    
    
    # output folder name: model + task_name + int differentiating between variants
    output_name = args.model
    if args.task_name:
        output_name += '_' + args.task_name
    output_folder = os.path.join(args.output_dir, output_name)
    output_int = 1
    while os.path.exists('{0}_{1}'.format(output_folder, output_int)):
        output_int += 1

    model_args.output_dir = '{0}_{1}'.format(output_folder, output_int)
    model_args.best_model_dir = os.path.join(model_args.output_dir, 'best_model')
    
    # path to trained LM
    lm_folder = os.path.join(args.language_model, args.lm_best_model_folder)
    
    # create and train the clf
    model = ClassificationModel(args.model, lm_folder, args=model_args, use_cuda=args.use_cuda)
    model.train_model(train_df=train_df, eval_df=train_df)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--data', nargs='*') # paths to txt-files with data
    arg_parser.add_argument('--delimiter', default='\t') # separates src from tgt in data file
    arg_parser.add_argument('--pairs', action='store_true') # classify src-tgt pairs; otherwise tgt is label
    arg_parser.add_argument('-m', '--model', default='bert') # model type (must be same as trained LM)
    arg_parser.add_argument('-lm', '--language_model') # path to trained LM
    arg_parser.add_argument('--lm_best_model_folder', default='best_model') # path to trained LM
    arg_parser.add_argument('--learning_rate', type=float, default=4e-05) # learning rate
    arg_parser.add_argument('--epochs', type=int, default=5) # number of training epochs
    arg_parser.add_argument('--vocab_size', type=int, default=30000) # max vocabulary size
    arg_parser.add_argument('--batch_size', type=int, default=1) # batch size
    arg_parser.add_argument('--output_dir', default='clf') # folder to save clf
    arg_parser.add_argument('--task_name', default='') # add to output folder name
    arg_parser.add_argument('--use_early_stopping', type=bool, default=False) # stop training when eval loss doesn't decrease
    arg_parser.add_argument('--random_seed', type=int, default=12345) # fix random seed
    arg_parser.add_argument('--use_cuda', action='store_true') # gpu/cpu
    args = arg_parser.parse_args()
    
    if args.random_seed:
        np.random.seed(args.random_seed)
    
    # create dataset from data files and train+save clf
    train_clf(args)
    
    # remove extra files created for training
    if os.path.exists('runs'):
        shutil.rmtree('runs')
    if os.path.exists('cache_dir'):
        shutil.rmtree('cache_dir')