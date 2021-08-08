import os
import json
import pandas as pd
from test_utils import *



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
    elif mode == 'label_only':
        test_df.columns = ["text", "labels"]
        tgt = [t for (s, t) in test_data]
    else:
        test_df.columns = ["input_text", "target_text"]
        tgt = [(s[0], t) for (s, t) in test_data] # return task label too

    return test_df, tgt
