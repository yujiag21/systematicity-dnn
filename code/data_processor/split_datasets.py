import os
import json
import argparse
import itertools
import collections
import numpy as np
import time


# specify dataset params and labels, create dataset, save to file
def main(args):
    
    ## read the process data file
    with open(args.input_processed_data_file, "r") as f:
        src_tgt = f.read().split("\n")
    
    np.random.shuffle(src_tgt)

    # save dataset parameters to json file
    params = {'task': args.task,
              'random_seed': args.random_seed,
              'size': len(src_tgt)}

    # make train-eval split
    train, val = [], []
    if args.eval_split:
        eval_split = round(args.eval_split * len(src_tgt))
        train, val = src_tgt[eval_split:], src_tgt[:eval_split]
        params['train_size'] = len(train)
        params['eval_size'] = len(val)
    
    # save folder name = task + vocabulary[:max_fname_len] + optional differentiating int
    save_folder = os.path.join(args.save_folder, args.task, args.dataset_name)

    if args.save_suffix:
        save_folder += '_' + args.save_suffix
    
    # add int to save_folder name for differentiating between variants
    if args.dont_overwrite:
        folder_int = 1
        while os.path.exists('{0}_{1}'.format(args.save_folder, folder_int)):
            folder_int += 1
        save_folder = '{0}_{1}'.format(save_folder, folder_int)
    
    print('Creating dataset for', save_folder, end=':')

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with open(os.path.join(save_folder, 'params.json'), 'w') as f:
        json.dump(params, f)

    # save data to a txt-file: tokens separated by separator, src and tgt separated by delimiter
    with open(os.path.join(save_folder, 'all.txt'), 'w') as f:
            f.write('\n'.join(src_tgt)) # add separators and delimiters

    if train != [] and val != []:
        with open(os.path.join(save_folder, 'train.txt'), 'w') as f:
            f.write('\n'.join(train))

        with open(os.path.join(save_folder, 'eval.txt'), 'w') as f:
            f.write("\n".join(val))

    print("finished...")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input_processed_data_file', type=str, required=True)
    arg_parser.add_argument('--dataset_name', type=str, required=True)
    arg_parser.add_argument('-task', '--task', required=True) # copy/different/reverse/count/repeat/custom
    arg_parser.add_argument('--random_seed', type=int, default=12345) # set random seed
    arg_parser.add_argument('--separator', default=' ') # separates tokens in saved txt-file
    arg_parser.add_argument('--delimiter', default='\t') # separates src from tgt in saved txt-file
    arg_parser.add_argument('--eval_split', type=float, default=0.2) # train-eval split
    arg_parser.add_argument('--save_folder', default='data/')
    arg_parser.add_argument('--save_suffix', default='') # add to save_folder name
    arg_parser.add_argument('--dont_overwrite', action='store_true') # add differentiating int to folder
    arg_parser.add_argument('--max_fname_len', type=int, default=100)
    args = arg_parser.parse_args()
    main(args)
