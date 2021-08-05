import os
import json
import pandas as pd
from clf_utils import *


def prepare_clf_data(classifier, data_files, pairs, delimiter):
    # get label names for tgt labels
    with open(os.path.join(classifier, 'label_names.json'), 'r') as f:
        label_names = json.load(f)

    clf_test_data = []

    for i, fpath in enumerate(data_files):
        data = open(fpath, 'r').readlines()
        data = [l.strip() for l in data]
        data = [l.split(delimiter) for l in data]

        # classifying pairs: format [src, tgt, label]; each file has a unique label
        if pairs:
            clf_test_data += [[s, t, i] for (s, t) in data]

        # classifying src to tgt: format [src, tgt]
        else:
            clf_test_data += [[s, label_names[t]] for (s, t) in data]

    test_df = pd.DataFrame(clf_test_data)

    if pairs:
        test_df.columns = ["text_a", "text_b", "labels"]
        tgt = [i for (s, t, i) in clf_test_data]
    else:
        test_df.columns = ["text", "labels"]
        tgt = [t for (s, t) in clf_test_data]

    return test_df, tgt