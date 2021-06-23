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
**custom**: map all sequences to a custom label given as the argument **--custom_label**\

Vocabulary given as a plaintext string: e.g. **ab**

## Training a language model (LM)

Create dataset (see above). Then run:

```bash
python train_lm -d *paths_to_datasets* -m *model_type*
```
