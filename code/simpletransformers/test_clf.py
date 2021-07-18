# Loading a trained classifier and evaluating it on test data

import os
import shutil
import pandas as pd
import argparse
from simpletransformers.classification import (ClassificationModel, ClassificationArgs)



def calculate_metrics_from_result(result):
    tp = result["tp"]
    tn = result["tn"]
    fp = result["fp"]
    fn = result["fn"]

    precision = tp/(tp+ fp)
    recall = tp / (tp +fn)

    f1 = tp / (tp + 0.5*(fp+fn))

    result["precision"] = precision
    result['recall'] = recall
    result['f1'] = f1

    return result

def test_trained_clf(args):
    
    clf_folder = os.path.join(args.classifier, args.clf_best_model_folder)
    
    model = ClassificationModel(args.model, clf_folder, use_cuda=args.use_cuda)
    
    # Make test data from original source files containing src-tgt pairs
    # CURRENTLY ONLY FOR --PAIRS; TODO FOR TASKS LIKE COUNT
    
    clf_test_data = []
    
    for i,fpath in enumerate(args.data):
        data = open(fpath, 'r').readlines()
        data = [l.strip() for l in data]
        data = [l.split(args.delimiter) for l in data]
        
        # classifying pairs: format [src, tgt, label]; each file has a unique label
        if args.pairs:
            clf_test_data += [[s,t,i] for (s,t) in data]
        
        # classifying src to tgt: format [src, tgt]
        # TODO
        else:
            print("RUN WITH --pairs!")
            return
    
    test_df = pd.DataFrame(clf_test_data)
    
    if args.pairs:
        test_df.columns = ["text_a", "text_b", "labels"]
    # else:
    #     test_df.columns = ["text", "labels"]
    
    result, outputs, wrong_predictions = model.eval_model(test_df)

    result = calculate_metrics_from_result(result)
    
    print(result)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--data', nargs='*') # paths to txt-files with test data
    arg_parser.add_argument('-m', '--model', default='bert') # model type (must be same as trained clf)
    arg_parser.add_argument('-clf', '--classifier', default='bert') # path to trained clf    
    arg_parser.add_argument('--delimiter', default='\t') # separates src from tgt in data file
    arg_parser.add_argument('--pairs', action='store_true') # classify src-tgt pairs; otherwise tgt is label
    arg_parser.add_argument('--clf_best_model_folder', default='best_model')
    arg_parser.add_argument('--results_dir', default='results') # folder to save results
    arg_parser.add_argument('--use_cuda', action='store_true') # gpu/cpu
    args = arg_parser.parse_args()
    
    test_trained_clf(args)
    
    # remove extra files
    if os.path.exists('runs'):
        shutil.rmtree('runs')
    if os.path.exists('cache_dir'):
        shutil.rmtree('cache_dir')
