import os
import json
import pandas as pd
from test_utils import *
import numpy as np
import edit_distance


TASK_LABEL = {
    "0": "different",
    "1": "copy",
    "2": "reverse"
}


def prepare_testing_data(classifier, data_files, mode, delimiter):
    # get label names for tgt labels

    assert mode in ['pairs', 'label_only', "enc_dec"]

    if mode != "enc_dec":
        with open(os.path.join(classifier, 'label_names.json'), 'r') as f:
            label_names = json.load(f)

    test_data = []

    for i, fpath in enumerate(data_files):
        data = open(fpath, 'r').readlines()
        data = [l.strip() for l in data]
        data = [l.split(delimiter) for l in data]

        # classifying pairs: format [src, tgt, label]; each file has a unique label
        if mode == 'pairs':
            test_data += [[s, t, i] for (s, t) in data]
        # classifying src to tgt: format [src, tgt]
        elif mode == 'label_only':
            test_data += [[s, label_names[t]] for (s, t) in data]
        else:
            test_data += data

    test_df = pd.DataFrame(test_data)

    if mode == 'pairs':
        test_df.columns = ["text_a", "text_b", "labels"]
        tgt = [i for (s, t, i) in test_data]
        src = [s for (s, t, i) in test_data]
    elif mode == 'label_only':
        test_df.columns = ["text", "labels"]
        tgt = [t for (s, t) in test_data]
        src = [s for (s, t) in test_data]
    else:
        test_df.columns = ["input_text", "target_text"]
        tgt = [t for (s, t) in test_data] # return task label too
        src = [s for (s, t) in test_data]

    return test_df, tgt, src



TASK_LABEL = {
    "0": "different",
    "1": "copy",
    "2": "reverse",
    "3": "dummy"
}


def calculate_metrics_enc_dec(tgts, srcs, preds):
    pairs = list(zip(tgts, srcs, preds))
    accuracy_arr = []
    accuracy_result = {}
    edit_distance_result = {}

    for tgt, src, pred in pairs:
        task = src[0]
        source_text = src[1:]
        target = tgt
        if task not in accuracy_result:
            accuracy_result[task] = []

        if task not in edit_distance_result:
            edit_distance_result[task] = []

        # different task
        if task == "0":
            acc = int(source_text != pred)
        else:
            acc = int(target == pred)

        accuracy_arr.append(acc)

        accuracy_result[task].append(acc)

        # calculate normalized edit distance
        edit_dist = edit_distance.SequenceMatcher(a=tgt, b=pred).distance() / max(len(tgt), len(pred))
        edit_distance_result[task].append(edit_dist)

    overall_acc = np.average(accuracy_arr)

    task_result = {'overall_acc': overall_acc}
    for key, val in accuracy_result.items():
        task_result[TASK_LABEL[key] + "-accuracy"] = np.average(val)

    for key, val in edit_distance_result.items():
        task_result[TASK_LABEL[key] + "-edit_distance"] = np.average(val)

    print(list(zip(srcs, preds)))

    return task_result, accuracy_arr, accuracy_result, edit_distance_result