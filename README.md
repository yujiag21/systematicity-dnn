# Systematic inference with DNNs

## Creating formal datasets

Run:

```bash
python create_datasets.py -task *task* -voc *vocabulary*
```

## Training a language model (LM)

Create dataset (see above). Then run:

```bash
python train_lm -d *paths_to_datasets* -m *model_type*
```
