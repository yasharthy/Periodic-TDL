# Periodic-TDL

Code and data release for **Periodic Topological Deep Learning for Polymer Design and Discovery**.

Periodic-TDL converts polymer pSMILES into periodic geometric representations, constructs periodic Vietoris-Rips filtrations over the repeat unit, and learns polymer representations with a Hierarchical Simplicial Message Passing (HSMP) encoder. The workflow is designed to retain covalent structure, periodic through-boundary proximity, multiscale topology, and higher-order simplex interactions before downstream property prediction.

## What Is Included

```text
code/       HSMP model, preprocessing scripts, pretraining, fine-tuning, and inference
data/       cleaned pretraining/fine-tuning data and acrylate Tg prediction tables
figures/    README figures for the repository, model, and data summaries
```

The main code workflow lives in [`code/`](code/). It covers periodic geometry construction, heterogeneous graph generation, self-supervised pretraining, downstream fine-tuning, checkpoint use, and Tg inference.

The released datasets are documented in [`data/`](data/). They include:

- a one-million-scale unlabeled pretraining set,
- downstream polymer property datasets with cross-validation folds,
- a generated 48,208-polymer acrylate/acrylamide library with Tg predictions, and
- a 22-polymer literature acrylate/acrylamide comparison set with Tg predictions.

Large model checkpoints are not committed to Git. Download links and placement instructions are provided in [`code/README.md`](code/README.md).

## Model Summary

Periodic-TDL starts from a polymer repeat unit and builds a periodic distance matrix that accounts for neighboring repeat units under periodic boundary conditions. Rips complexes at multiple cutoffs are then packed as PyTorch Geometric `HeteroData` graphs.

HSMP performs message passing within each filtration level and cross-scale refinement from coarser to finer cutoffs, allowing atom-scale representations to incorporate long-range and higher-order topological context.

The released workflow supports two main paths:

1. use pretrained HSMP checkpoints for downstream fine-tuning or Tg inference;
2. rebuild the periodic complexes and pretrain HSMP from scratch.

## Environment and Installation

The code was tested on **NSCC Singapore ASPIRE 2A**, using:

- Red Hat Enterprise Linux 8
- NVIDIA A100 40 GB GPU
- CUDA 11.6
- Python 3.10
- PyTorch 1.13.1 + CUDA 11.6
- PyTorch Geometric 2.6.1

A CUDA-enabled GPU is required for model training.

## Quick Demo and Reproduction

For a lightweight test of the downstream pipeline, we recommend the **Eea electron-affinity dataset**, which contains 368 polymers.

After downloading the pretrained checkpoints, run one fold from the `code/` directory:

```bash
python 1_periodic_geom.py --datafile Eea

python 2_build_complex_finetune.py \
    --datafile Eea \
    --procs 16

python 5_downstream.py \
    --datafile Eea \
    --split 0 \
    --ckpt pretrain_1M_768D12H/encoder_pretrained.best.pt \
    --heads 12 \
    --epochs 60
```

The downstream script reports RMSE and R² during training/validation and reports the corresponding metrics on the held-out test set after training.

A single Eea fold should complete in **less than one hour** on the tested hardware under typical conditions. Pretraining the full model from scratch should complete in **less than one day**.

For downstream experiments, the released `<dataset>_folds.pkl` files define the fixed train/test splits used in the study. Within each training fold, the training data are further divided into training and validation sets. Detailed commands for preprocessing, pretraining, fine-tuning all downstream datasets, checkpoint use, inference, and running Periodic-TDL on custom data are provided in [`code/README.md`](code/README.md).
