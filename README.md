# Systematic inference with DNNs

## Setup

Install the simpletransformers library (anaconda/miniconda recommended):\
https://simpletransformers.ai/

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

By default, dataset is saved to a folder in "data/", with the name "*task*_*vocabulary[:5]*_*int*.txt"\
The int is to differentiate datasets with the same parameters to avoid overwriting.\
Saving options can be modified via the arguments **--save_folder**, **--params_fname**, and **data_fname**.

## Training a language model (LM)

Create dataset (see above). Then run:

```bash
python train_lm -d *paths_to_datasets* -m *model_type* (--pairs)
```

Multiple datasets can be given as argument to -d. The example below first creates two datasets and then trains a LM based on them (default settings for dataset names based on task & vocabulary):

```bash
python create_datasets.py -task copy -voc ab
python create_datasets.py -task different -voc cd

python train_lm -d data/copy_ab_1/src_tgt.txt data/different_cd_1/src_tgt.txt -m bert --pairs
```

Adding --pairs takes both src and tgt from the original data file into consideration when forming the LM training data.
Not adding it only takes the src.

By default, the LM training data is saved in the folder lm/lm_training_data, and named as vocabulary[:5]_*int*, similarly to the original dataset (see above).

