# Fine-tuning a LM to sequence-to-sequence mapping
# Encoder must be in [bert, roberta, distilbert, camembert, electra]
# Decoder must be bert

import os
import json
import shutil
import logging
import argparse
import numpy as np
import pandas as pd
from simpletransformers.seq2seq import (Seq2SeqModel, Seq2SeqArgs)


def train_clf(args):
    
    # Make training data from original source files containing src-tgt pairs
    
    train_data = []
    
    for i,fpath in enumerate(args.train_data):
        data = open(fpath, 'r').readlines()
        data = [l.strip() for l in data]
        data = [l.split(args.delimiter) for l in data]
        train_data += data
        
    np.random.shuffle(train_data)
    train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])
    
    eval_data = []
    eval_fpaths = args.eval_data if args.eval_data else []
    
    for i,fpath in enumerate(eval_fpaths):
        data = open(fpath, 'r').readlines()
        data = [l.strip() for l in data]
        data = [l.split(args.delimiter) for l in data]
        eval_data += data
        
    eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"]) if eval_data else train_df

    model_args = Seq2SeqArgs()
    model_args.overwrite_output_dir = True
    model_args.reprocess_input_data = True
    model_args.evaluate_generated_text = True
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
    
    # output folder name: model + task_name + optional int differentiating between variants
    output_name = args.model
    if args.task_name:
        output_name += '_' + args.task_name
    output_folder = os.path.join(args.output_dir, output_name)
    
    if args.dont_overwrite:
        output_int = 1
        while os.path.exists('{0}_{1}'.format(output_folder, output_int)):
            output_int += 1
        output_folder = '{0}_{1}'.format(output_folder, output_int)
        
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)    

    model_args.output_dir = output_folder
    model_args.best_model_dir = os.path.join(model_args.output_dir, 'best_model')

    # path to trained LM
    lm_folder = os.path.join(args.language_model, args.lm_best_model_folder)
    
    # create and train the clf
    model = Seq2SeqModel(encoder_type=args.model,
                         encoder_name=lm_folder,
                         decoder_name=lm_folder,
                         args=model_args,
                         use_cuda=args.use_cuda)
    
    model.train_model(train_data=train_df, eval_data=eval_df)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--train_data', nargs='*') # paths to txt-files with training data
    arg_parser.add_argument('--eval_data', nargs='*') # paths to txt-files with evaluation data
    arg_parser.add_argument('--delimiter', default='\t') # separates src from tgt in data file
    arg_parser.add_argument('-m', '--model', default='bert') # model type for the encoder: bert/roberta/distilbert/camembert/electra
    arg_parser.add_argument('-lm', '--language_model') # path to trained LM
    arg_parser.add_argument('--lm_best_model_folder', default='best_model') # path to trained LM
    arg_parser.add_argument('--learning_rate', type=float, default=4e-05) # learning rate
    arg_parser.add_argument('--epochs', type=int, default=5) # number of training epochs
    arg_parser.add_argument('--vocab_size', type=int, default=30000) # max vocabulary size
    arg_parser.add_argument('--batch_size', type=int, default=16) # batch size
    arg_parser.add_argument('--output_dir', default='enc_dec') # folder to save encoder-decoder model
    arg_parser.add_argument('--task_name', default='') # add to output folder name
    arg_parser.add_argument('--use_early_stopping', action='store_true') # stop training when eval loss doesn't decrease
    arg_parser.add_argument('--random_seed', type=int, default=12345) # fix random seed
    arg_parser.add_argument('--dont_overwrite', action='store_true') # add differentiating int to folder
    arg_parser.add_argument('--use_cuda', action='store_true') # gpu/cpu
    args = arg_parser.parse_args()
    
    if args.random_seed:
        np.random.seed(args.random_seed)
    
    # create dataset from data files and train+save enc-dec model
    train_clf(args)
    
    # remove extra files
    if os.path.exists('runs'):
        shutil.rmtree('runs')
    if os.path.exists('cache_dir'):
        shutil.rmtree('cache_dir')