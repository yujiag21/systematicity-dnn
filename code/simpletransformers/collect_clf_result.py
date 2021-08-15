import os
import json
import shutil
import pandas as pd
import argparse
from sklearn.metrics import precision_recall_fscore_support
from simpletransformers.classification import (ClassificationModel, ClassificationArgs)
from test_utils import *
import re

MODEL_DIR_TEMPLATE = 'checkpoint-[0-9]+-epoch-'
RESULT_COLUMN_NAMES = ["epoch", "dir_name", "eval_loss", "precision", "recall", "f1"]


def test_trained_clf(args, output_filepath):

    test_result = {
        "epoch": [],
        "dir_name": [],
        "eval_loss": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    # prepare all test data and label

    mode = 'pairs' if args.pairs else 'label_only'
    test_df, tgt, _ = prepare_testing_data(args.classifier, args.data, mode, args.delimiter)

    # collect all the models at different epoch
    dirs = [d for d in os.listdir(args.classifier) if os.path.isdir(os.path.join(args.classifier, d))]
    epoch_dir = map(lambda x: (int(re.sub(MODEL_DIR_TEMPLATE, '', x)), x),
                filter(lambda x: re.match(MODEL_DIR_TEMPLATE, x), dirs))

    # run experiment over models at different epoch on test dataset
    for epoch, dir_name in epoch_dir:

        clf_folder = os.path.join(args.classifier, dir_name)
        model = ClassificationModel(args.model, clf_folder, use_cuda=args.use_cuda)

        result, outputs, wrong_predictions = model.eval_model(test_df)
        preds = [list(l).index(max(l)) for l in outputs]

        precision, recall, f1, support = precision_recall_fscore_support(y_true=tgt, y_pred=preds, average='macro',
                                                                         zero_division=0)

        result["precision"] = precision
        result['recall'] = recall
        result['f1'] = f1

        test_result["epoch"].append(epoch)
        test_result["dir_name"].append(dir_name)
        test_result['eval_loss'].append(result['eval_loss'])
        test_result["precision"].append(result["precision"])
        test_result["recall"].append(result["recall"])
        test_result["f1"].append(result["f1"])

    test_result_df = pd.DataFrame(test_result, columns=RESULT_COLUMN_NAMES)
    test_result_df.to_csv(output_filepath, index=False)




if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--data', nargs='*')  # paths to txt-files with test data
    arg_parser.add_argument("-dl", "--data_label", default='test',
                            help='specify whether this data is from test data or evaluation data')
    arg_parser.add_argument('-m', '--model', default='bert')  # model type (must be same as trained clf)
    arg_parser.add_argument('-clf', '--classifier', default='bert')  # path to trained clf
    arg_parser.add_argument('--delimiter', default='\t')  # separates src from tgt in data file
    arg_parser.add_argument('--pairs', action='store_true')  # classify src-tgt pairs; otherwise tgt is label
    arg_parser.add_argument('--clf_model_folder', default='clf')
    arg_parser.add_argument('--use_cuda', action='store_true')  # gpu/cpu
    arg_parser.add_argument('--results_dir', default='results')  # folder to save results
    arg_parser.add_argument('-rf','--result_filename', default='')
    args = arg_parser.parse_args()

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    if not args.result_filename:
        output_filepath = os.path.join(args.results_dir, args.data_label + "_metrics.csv")
    else:
        output_filepath = os.path.join(args.results_dir, args.result_filename)

    test_trained_clf(args, output_filepath)

    # remove extra files
    if os.path.exists('runs'):
        shutil.rmtree('runs')
    if os.path.exists('cache_dir'):
        shutil.rmtree('cache_dir')
