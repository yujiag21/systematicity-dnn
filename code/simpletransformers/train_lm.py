# Training language model (LM)
# Takes txt-files with src-tgt pairs as arguments, pools together and trains LM
# Argument files created with create_datasets.py

import os
import shutil
import logging
import argparse
import numpy as np
from simpletransformers.language_modeling import (LanguageModelingModel, LanguageModelingArgs)

# prevents unnecessary warning message
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train_lm(args):
    
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
    
    if not os.path.exists(args.save_data_dir):
        os.makedirs(args.save_data_dir)
    
    # filepath to save training data: named by vocabulary[:5] + added int for differentiating between variants
    cont_str = '...' if len(lm_train_voc) > 5 else ''
    data_int = 1
    train_fname = ''.join(lm_train_voc[:5]) + cont_str
    while os.path.exists('{0}_{1}.txt'.format(train_fname, data_int)):
        data_int += 1
    train_file = '{0}_{1}.txt'.format(train_fname, data_int)
    # train_file = ''.join(lm_train_voc[:5]) + cont_str + '.txt'
    train_file = os.path.join(args.save_data_dir, train_file)
    
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
    model_args.overwrite_output_dir = True
    model_args.reprocess_input_data = True
    model_args.evaluate_during_training = True
    model_args.early_stopping_consider_epochs = True
    model_args.dataset_type = "simple"
    model_args.early_stopping_patience = 1
    
    model_args.num_train_epochs = args.epochs
    model_args.learning_rate = args.learning_rate
    model_args.vocab_size = args.vocab_size
    model_args.train_batch_size = args.batch_size
    model_args.eval_batch_size = args.batch_size
    model_args.use_early_stopping = args.use_early_stopping
    
    # output folder name: model + vocabulary + int differentiating between variants
    output_name = '{0}_{1}'.format(args.model, train_fname)
    output_folder = os.path.join(args.output_dir, output_name)
    output_int = 1
    while os.path.exists('{0}_{1}'.format(output_folder, output_int)):
        output_int += 1
    
    model_args.output_dir = '{0}_{1}'.format(output_folder, output_int)
    model_args.best_model_dir = os.path.join(model_args.output_dir, 'best_model')
    
    model = LanguageModelingModel(args.model, None, args=model_args, train_files=train_file, use_cuda=args.use_cuda)
    
    # train model
    model.train_model(train_file, eval_file=train_file)
    result = model.eval_model(train_file)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--data', nargs='*') # paths to txt-files with data
    arg_parser.add_argument('--delimiter', default='\t') # separates src from tgt in data file
    arg_parser.add_argument('--only_src', action='store_true') # use only src from data file for training
    arg_parser.add_argument('--save_data_dir', default='lm/lm_training_data') # folder to save LM training data
    arg_parser.add_argument('-m', '--model', default='bert') # model type
    arg_parser.add_argument('--learning_rate', type=float, default=4e-05) # learning rate
    arg_parser.add_argument('--epochs', type=int, default=5) # number of training epochs
    arg_parser.add_argument('--vocab_size', type=int, default=30000) # max vocabulary size
    arg_parser.add_argument('--batch_size', type=int, default=1) # batch size
    arg_parser.add_argument('--output_dir', default='lm') # folder to save LM
    arg_parser.add_argument('--use_early_stopping', type=bool, default=True) # stop training when eval loss doesn't decrease
    arg_parser.add_argument('--random_seed', type=int, default=12345) # fix random seed
    arg_parser.add_argument('--use_cuda', action='store_true') # gpu/cpu
    args = arg_parser.parse_args()
    
    if args.random_seed:
        np.random.seed(args.random_seed)
    
    # create dataset from data files and train+save LM
    train_lm(args)
    
    # remove extra files created for training
    if os.path.exists('runs'):
        shutil.rmtree('runs')
    if os.path.exists('cache_dir'):
        shutil.rmtree('cache_dir')