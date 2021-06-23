# Systematic inference with DNNs

## Setup

Install the simpletransformers library (anaconda/miniconda recommended):\
https://simpletransformers.ai/

All code is in the folder *code*\
The files train_lm.py and train_clf.py are in the folder *code/simpletransformers*.

## Creating formal language datasets

Run:

```bash
python create_datasets.py -task *task* -voc *vocabulary*
```

Task is one of the following:\
**copy**: map sequences to themselves\
**reverse**: map sequences to their reversal\
**uppercase**: map sequences to capitalized versions\
**count**: map sequences to their token counts represented by a single number token\
**custom**: map all sequences to a custom label given as the argument **--custom_label**

Vocabulary given as a plaintext string: e.g. **ab**

Some examples:

```bash
python create_datasets.py -task copy -voc ab
python create_datasets.py -task different -voc cd
python create_datasets.py -task custom -voc abc --custom_label X
```

By default, dataset is saved to *data/task_vocabulary[:5]_int/src_tgt.txt*\
The int is to differentiate datasets with the same parameters to avoid overwriting.\
Saving options can be modified via the arguments **--save_folder**, **--params_fname**, and **data_fname**.

## Training a language model (LM)

Create dataset (see above). Then run:

```bash
python train_lm.py -d *paths_to_datasets* -m *model_type* (--pairs)
```

Multiple datasets can be given as argument to **-d**.\
The training data is constructed from all arguments of **-d**.

Adding **--pairs** takes both src and tgt from the original data file into consideration when forming the LM training data.\
Not adding it only uses the src for training the LM.

By default, the LM training data is saved in the folder *lm/lm_training_data*, and named as *vocabulary[:5]_int.txt*\
Training data saving location be modified via the argument **--save_data_dir**.

By default, the LM is saved in the folder "lm", and named as *model_vocabulary[:5]_int.txt*\
LM saving location can be modified via the argument **--output_dir**.

See simpletransformers documentation (above) for list of available LM types (e.g. BERT, RoBERTa, etc.).

Using cpu by default; switch to gpu with **--use_cuda**.

## Training a classifier (clf)

Create dataset and train a LM on it (see above). Then run:

```bash
python train_clf.py -d *paths_to_datasets* -m *model_type* (--pairs)
```

Adding **--pairs** classifier sentence pairs rather than single sentences. Here, the label is based on the data file (the first **-d** argument gets the label 0, the second gets 1, etc.). Without **--pairs** the label is taken from tgt.

Using cpu by default; switch to gpu with **--use_cuda**.

## Full pipeline for training clf from scratch

The example below does the following (default settings for dataset names based on task & vocabulary):\
1. creates two datasets
2. trains a LM from the datapoints of the datasets (both src and tgt here due to **--pairs**)
3. trains a clf from the same datasets, using the trained LM as the base model

```bash
python create_datasets.py -task copy -voc ab
python create_datasets.py -task different -voc cd

python simpletransformers/train_lm.py -d data/copy_ab_1/src_tgt.txt data/different_cd_1/src_tgt.txt -m bert --pairs

python simpletransformers/train_clf.py -d data/copy_ab_1/src_tgt.txt data/different_cd_1/src_tgt.txt -m bert -lm lm/bert_abcd_1 --pairs
```

