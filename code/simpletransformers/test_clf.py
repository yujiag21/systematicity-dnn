# Loading a trained classifier and evaluating it on test data

import os
import json
import shutil
import pandas as pd
import argparse
from sklearn.metrics import precision_recall_fscore_support
from simpletransformers.classification import (ClassificationModel, ClassificationArgs)
from clf_utils import *

def test_trained_clf(args):
    
    clf_folder = os.path.join(args.classifier, args.clf_best_model_folder)
    model = ClassificationModel(args.model, clf_folder, use_cuda=args.use_cuda)
    
    clf_test_data = []
    
    for i,fpath in enumerate(args.data):
        data = open(fpath, 'r').readlines()
        data = [l.strip() for l in data]
        data = [l.split(args.delimiter) for l in data]
        
        # classifying pairs: format [src, tgt, label]; each file has a unique label
        if args.pairs:
            clf_test_data += [[s,t,i] for (s,t) in data]
        
        # classifying src to tgt: format [src, tgt]
        else:
            clf_test_data += [[s, label_names[t]] for (s,t) in data]
    
    test_df = pd.DataFrame(clf_test_data)
    
    if args.pairs:
        test_df.columns = ["text_a", "text_b", "labels"]
        tgt = [i for (s, t, i) in clf_test_data]
    else:
        test_df.columns = ["text", "labels"]
        tgt = [t for (s,t) in clf_test_data]
    
    result, outputs, wrong_predictions = model.eval_model(test_df)
    preds = [list(l).index(max(l)) for l in outputs]
    precision, recall, f1, support = precision_recall_fscore_support(y_true=tgt, y_pred=preds, average='macro', zero_division=0)
    
    result['precision'] = precision
    result['recall'] = recall
    result['f1'] = f1

    print(result)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--data', nargs='*') # paths to txt-files with test data
    arg_parser.add_argument('-m', '--model', default='bert') # model type (must be same as trained clf)
    arg_parser.add_argument('-clf', '--classifier', default='bert') # path to trained clf    
    arg_parser.add_argument('--delimiter', default='\t') # separates src from tgt in data file
    arg_parser.add_argument('--pairs', action='store_true') # classify src-tgt pairs; otherwise tgt is label
    arg_parser.add_argument('--clf_best_model_folder', default='best_model',
                            help="Name of the directory for the version of model used. "
                                 "You can use an older model version from earlier epoch. "
                                 "(Default is 'best_model')")
    arg_parser.add_argument('--results_dir', default='results') # folder to save results
    arg_parser.add_argument('--use_cuda', action='store_true') # gpu/cpu
    args = arg_parser.parse_args()
    
    test_trained_clf(args)
    
    # remove extra files
    if os.path.exists('runs'):
        shutil.rmtree('runs')
    if os.path.exists('cache_dir'):
        shutil.rmtree('cache_dir')
