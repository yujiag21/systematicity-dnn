# Loading a trained classifier and evaluating it on test data

import os
import json
import shutil
import pandas as pd
import argparse
from sklearn.metrics import precision_recall_fscore_support
from simpletransformers.seq2seq import (Seq2SeqModel, Seq2SeqArgs)
from test_utils import *
import numpy as np


TASK_LABEL = {
    "0": "different",
    "1": "copy",
    "2": "reverse",
    "3": "dummy"
}


def accuracy_measure_seq(tgts, preds):
    pairs = list(zip(tgts, preds))
    result_arr = []
    task_result_result = {"0":[], "1":[], "2":[], "3": []}

    for tgt, pred in pairs:
        if tgt[0] == "0":
            res = int(tgt[1] != pred)
            result_arr.append(res)
        else:
            res = int(tgt[1] == pred)
            result_arr.append(res)

        task_result_result[tgt[0]].append(res)


    overall_acc = np.average(result_arr)

    task_acc = {'overall_acc': overall_acc}
    for key, val in task_result_result.items():
        if val:
            task_acc[TASK_LABEL[key]] = np.average(val)

    return task_acc, result_arr, task_result_result



def test_trained_encoder_decoder(args):
    enc_folder = os.path.join(args.encoder_decoder, args.enc_dec_best_model_folder, "encoder")
    dec_folder = os.path.join(args.encoder_decoder, args.enc_dec_best_model_folder, "decoder")

    model = Seq2SeqModel(args.model, encoder_name=enc_folder, decoder_name=dec_folder, use_cuda=args.use_cuda)

    # prepare all test data and label
    test_df, tgt = prepare_testing_data(args.encoder_decoder, args.data, "enc_dec", args.delimiter)
    preds = model.predict(test_df['input_text'].to_list())

    task_acc, result_arr, task_result_result = accuracy_measure_seq(tgt, preds)

    print(task_acc)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--data', nargs='*')  # paths to txt-files with test data
    arg_parser.add_argument('-m', '--model', default='bert')  # model type (must be same as trained clf)
    arg_parser.add_argument('-enc_dec', '--encoder_decoder', default='bert')  # path to trained clf
    arg_parser.add_argument('--delimiter', default='\t')  # separates src from tgt in data file
    arg_parser.add_argument('--enc_dec_best_model_folder', default='best_model',
                            help="Name of the directory for the version of model used. "
                                 "You can use an older model version from earlier epoch. "
                                 "(Default is 'best_model')")
    arg_parser.add_argument('--results_dir', default='results')  # folder to save results
    arg_parser.add_argument('--use_cuda', action='store_true')  # gpu/cpu
    args = arg_parser.parse_args()

    test_trained_encoder_decoder(args)

    # remove extra files
    if os.path.exists('runs'):
        shutil.rmtree('runs')
    if os.path.exists('cache_dir'):
        shutil.rmtree('cache_dir')
