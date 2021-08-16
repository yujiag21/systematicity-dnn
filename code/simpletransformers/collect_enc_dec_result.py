import os
import json
import shutil
import pandas as pd
import argparse

from simpletransformers.seq2seq import Seq2SeqModel
from sklearn.metrics import precision_recall_fscore_support
from simpletransformers.classification import (ClassificationModel, ClassificationArgs)
from test_utils import *
import re

MODEL_DIR_TEMPLATE = 'checkpoint-[0-9]+-epoch-'


def test_trained_enc_dec(args, output_filepath):

    test_result = {
        "epoch": [],
        "dir_name": []
    }

    # prepare all test data and label

    mode = 'enc_dec'
    test_df, tgt, src = prepare_testing_data(args.encoder_decoder, args.data, mode, args.delimiter)

    # collect all the models at different epoch
    dirs = [d for d in os.listdir(args.encoder_decoder) if os.path.isdir(os.path.join(args.encoder_decoder, d))]
    epoch_dir = map(lambda x: (int(re.sub(MODEL_DIR_TEMPLATE, '', x)), x),
                filter(lambda x: re.match(MODEL_DIR_TEMPLATE, x), dirs))

    # run experiment over models at different epoch on test dataset
    for epoch, dir_name in epoch_dir:

        enc_folder = os.path.join(args.encoder_decoder, dir_name, "encoder")
        dec_folder = os.path.join(args.encoder_decoder, dir_name, "decoder")

        model = Seq2SeqModel(args.model, encoder_name=enc_folder, decoder_name=dec_folder, use_cuda=args.use_cuda)

        preds = model.predict(test_df['input_text'].to_list())

        task_result, accuracy_arr, accuracy_result, edit_distance_result = calculate_metrics_enc_dec(tgt, src, preds)

        test_result["epoch"].append(epoch)
        test_result["dir_name"].append(dir_name)

        for key, val in task_result.items():
            if key not in test_result:
                test_result[key] = []

            test_result[key].append(val)

    test_result_df = pd.DataFrame(test_result, columns=test_result.keys())
    test_result_df.to_csv(output_filepath, index=False)




if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--data', nargs='*')  # paths to txt-files with test data
    arg_parser.add_argument("-dl", "--data_label", default='test',
                            help='specify whether this data is from test data or evaluation data')
    arg_parser.add_argument('-m', '--model', default='bert')  # model type (must be same as trained clf)
    # arg_parser.add_argument('-clf', '--classifier', default='bert')  # path to trained clf
    arg_parser.add_argument('-enc_dec', '--encoder_decoder', default='bert')
    arg_parser.add_argument('--delimiter', default='\t')  # separates src from tgt in data file
    arg_parser.add_argument('--enc_dec_model_folder', default='enc_dec')
    arg_parser.add_argument('--use_cuda', action='store_true')  # gpu/cpu
    arg_parser.add_argument('--results_dir', default='results')  # folder to save results
    arg_parser.add_argument('-rf', '--result_filename', default='')
    args = arg_parser.parse_args()

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    if not args.result_filename:
        output_filepath = os.path.join(args.results_dir, args.data_label + "_metrics.csv")
    else:
        output_filepath = os.path.join(args.results_dir, args.result_filename)

    test_trained_enc_dec(args, output_filepath)

    # remove extra files
    if os.path.exists('runs'):
        shutil.rmtree('runs')
    if os.path.exists('cache_dir'):
        shutil.rmtree('cache_dir')
