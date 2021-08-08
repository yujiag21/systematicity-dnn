# Loading a trained classifier and evaluating it on test data

import os
import json
import shutil
import pandas as pd
import argparse
from sklearn.metrics import precision_recall_fscore_support
from simpletransformers.classification import (ClassificationModel, ClassificationArgs)
from test_utils import *


def test_trained_clf(args):
    
    clf_folder = os.path.join(args.classifier, args.clf_best_model_folder)
    model = ClassificationModel(args.model, clf_folder, use_cuda=args.use_cuda)

    # prepare all test data and label
    mode = 'pairs' if args.pairs else 'label_only'
    test_df, tgt = prepare_testing_data(args.classifier, args.data, mode, args.delimiter)

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
