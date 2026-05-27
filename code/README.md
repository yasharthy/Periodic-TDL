# Periodic-TDL

Code release for **Periodic Topological Deep Learning for Polymer Design and
Discovery**.

Periodic-TDL represents linear homopolymers with a periodic Vietoris-Rips
filtration constructed from a periodic distance matrix. This encodes interactions
between atoms across adjacent repeating units under periodic boundary conditions,
including many-body interactions through higher-dimensional simplices. The
Hierarchical Simplicial Message Passing (HSMP) encoder then performs multi-head
simplicial message passing within each filtration level and cross-scale
refinement from coarser to finer spatial scales. Atom representations at the
covalent-bond scale therefore incorporate long-range and higher-order
topological information.

Simplex features combine RDKit atom/bond descriptors with geometry-aware
Forman-Ricci curvature features, following the Periodic-TDL paper workflow.

![Figure 2: HSMP encoder, cross-scale refinement, and Periodic-TDL training workflow.](../figures/Fig1.png)

This repository contains the scripts needed to:

1. construct periodic Rips complexes from pSMILES,
2. build nested sequences of Rips complexes using PyTorch Geometric `HeteroData`,
3. pretrain HSMP on one million unlabelled polymers,
4. fine-tune Periodic-TDL on nine labelled polymer property datasets, and
5. apply trained deployment models to generated acrylate/acrylamide polymer libraries.

The README describes two use cases:

1. use pretrained checkpoints,
2. pretrain from scratch.

## Checkpoints

Large checkpoint files are not committed to this repository. 
Pretrained encoder checkpoints and deployment checkpoints are available on Google Drive.

[Download checkpoints from Google Drive](https://drive.google.com/drive/folders/1p5nKNh1MVa8kU8oTx5EdWXlYDycuGOOv?usp=drive_link)

After downloading the checkpoints, place or extract them so the `code/` folder
contains the following paths.

Pretrained encoder checkpoints:
```text
ckpt_pretrain/pretrain_1M_768D01H/encoder_pretrained.best.pt
ckpt_pretrain/pretrain_1M_768D03H/encoder_pretrained.best.pt
ckpt_pretrain/pretrain_1M_768D06H/encoder_pretrained.best.pt
ckpt_pretrain/pretrain_1M_768D12H/encoder_pretrained.best.pt
```

Deployment checkpoints used by `6_predict_PI1M.py`:
```text
ckpt_deploy/Tg/split_0/finetuned.best.pt
...
ckpt_deploy/Tg/split_9/finetuned.best.pt
```

Intermediate fine-tuning checkpoints are not distributed. Running `5_downstream.py` creates local outputs under `ckpt_finetune/` by default.

Checkpoint naming convention:
- `768D`: hidden dimension 768.
- `01H`, `03H`, `06H`, `12H`: number of simplicial message-passing heads.

The model architecture must match the checkpoint. For example, use `--heads 12` with `pretrain_1M_768D12H`.

## Repository Layout
```text
|-- 1_periodic_geom.py              # periodic distance/geometric construction
|-- 2_build_complex_pretrain.py     # sharded HeteroData graphs for pretraining
|-- 2_build_complex_finetune.py     # labelled HeteroData graphs for downstream tasks
|-- 3_build_vocab.py                # atom/bond context vocabularies
|-- 4_pretrain.py                   # self-supervised HSMP pretraining
|-- 5_downstream.py                 # downstream fine-tuning and evaluation
|-- 6_predict_PI1M.py               # Tg prediction using deployment checkpoints
|-- models/                         # HSMP encoder and task heads
|-- utils/                          # chemistry, geometry, graph, label, and packing utilities
|-- data/                           # generated intermediate data structures
|-- ckpt_pretrain/                  # populated after checkpoint download
|-- ckpt_deploy/                    # populated after checkpoint download
```

The release input datasets live outside this folder at `../data/`; see
[`../data/README.md`](../data/README.md) for dataset details.
## Environment
The archived runs used Python 3.10 with CUDA-enabled PyTorch. Core dependencies
include:

- `torch`
- `torch-geometric`
- `rdkit`
- `gudhi`
- `networkx`
- `numpy`
- `pandas`
- `scikit-learn`
- `tqdm`
- `psutil`

## Data Format
The scripts expect processed data under:

```text
../data/processed/
```

For a dataset basename `<name>`, the expected files are:

```text
../data/processed/<name>_cleaned.csv
../data/processed/<name>_folds.pkl       # downstream fine-tuning only
```

Required CSV columns:

- `smiles`: polymer pSMILES string.
- `value`: scalar target value for downstream fine-tuning datasets.

`<name>_folds.pkl` should be a pickled list of fold dictionaries. Each fold must
contain row-index lists named `train` and `test`.

The nine downstream property basenames are:

| Basename | Property | Class | Unit |
| --- | --- | --- | --- |
| `Egc` | Bandgap (chain) | Electronic | eV |
| `Eib` | Electron injection barrier | Electronic | eV |
| `Egb` | Bandgap (bulk) | Electronic | eV |
| `Eea` | Electron affinity | Electronic | eV |
| `Ei` | Ionization energy | Electronic | eV |
| `EPS` | Dielectric constant | Optical | dimensionless |
| `Nc` | Refractive index | Optical | dimensionless |
| `Xc` | Crystallization tendency | Physical | percent |
| `Tg` | Glass transition temperature | Thermal | deg C |

`Tg` is experimentally measured; the remaining properties are DFT-derived.

## Option 1: Use Pretrained Checkpoints

Download the checkpoint bundle from the Google Drive link above and place the
pretrained encoders under `ckpt_pretrain/`.

### Build Downstream Graphs

For each property dataset, first build periodic geometries and labelled
heterogeneous graphs:

```bash
python 1_periodic_geom.py --datafile Tg
python 2_build_complex_finetune.py --datafile Tg --procs 16
```

This writes:

```text
data/Coordinates/Tg_periodic.pkl
data/HeteroGraphsDownstream/Tg_hetero.pt
```

### Fine-Tune From a Pretrained Encoder

`5_downstream.py` resolves `--ckpt` relative to `ckpt_pretrain/`. It loads the
checkpoint metadata and initializes the HSMP encoder from `ckpt["encoder"]`:

```python
if "encoder" in ckpt:
    model.embedder.load_state_dict(ckpt["encoder"], strict=False)
```

Example for `Tg`, split 0, using the 12-head checkpoint:

```bash
python 5_downstream.py \
  --datafile Tg \
  --split 0 \
  --ckpt pretrain_1M_768D12H/encoder_pretrained.best.pt \
  --heads 12 \
  --batch_size 64 \
  --epochs 60 \
  --emb_drop 0.0 \
  --reg_drop 0.0 \
  --lr_encoder 0.0001 \
  --lr_head 0.001 \
  --warmup_epochs 10 \
  --T0 10
```

Outputs are written to `ckpt_finetune/` unless `--outdir` is changed. These
fine-tuning outputs are local experiment artifacts and are not part of the public
checkpoint bundle.

### Run All Nine Downstream Tasks

The archived runs used property-specific batch sizes and regularization. A
minimal loop is:

```bash
DATASETS=(Tg Egc Eib Egb Eea Ei EPS Nc Xc)

for ds in "${DATASETS[@]}"; do
  for split in {0..4}; do
    python -u 5_downstream.py \
      --datafile "$ds" \
      --split "$split" \
      --ckpt pretrain_1M_768D12H/encoder_pretrained.best.pt \
      --heads 12 \
      --emb_drop 0.0 \
      --epochs 60 \
      --warmup_epochs 10 \
      --T0 10
  done
done
```

Adjust `--batch_size`, `--weight_decay`, and `--reg_drop` for each property and
available GPU memory.

## Option 2: Pretrain From Scratch

Pretraining follows the self-supervised pipeline described in the paper:
atom context prediction, bond context prediction, and functional group
prediction. The expected pretraining basename in the scripts is `pretrain_1M`.

### 1. Build Periodic Geometries

```bash
python 1_periodic_geom.py --datafile pretrain_1M
```

Input:

```text
../data/processed/pretrain_1M_cleaned.csv
```

Output:

```text
data/Coordinates/pretrain_1M_periodic.pkl
```

### 2. Build Sharded Heterogeneous Graphs

For 1M-scale pretraining, save graphs in chunks so `4_pretrain.py` can stream
them from disk:

```bash
python 2_build_complex_pretrain.py \
  --datafile pretrain_1M \
  --procs 64 \
  --chunks 100
```

For scratch pretraining, `3_build_vocab.py` and `4_pretrain.py` expect:

```text
data/HeteroGraphsPretrain/pretrain_1M_v1/
```

Check `OUTDIR_DEFAULT` in `2_build_complex_pretrain.py` before launching graph
construction. Expected shard names are:

```text
data/HeteroGraphsPretrain/pretrain_1M_v1/pretrain_1M_hetero_part00.pt
data/HeteroGraphsPretrain/pretrain_1M_v1/pretrain_1M_hetero_part01.pt
...
```

### 3. Build Context Vocabularies

```bash
python 3_build_vocab.py \
  --datafile pretrain_1M \
  --num_chunks 100 \
  --max_workers 50
```

Outputs:

```text
data/CtxVocab/atom_vocab.json
data/CtxVocab/bond_vocab.json
```

### 4. Pretrain HSMP

Example matching the 12-head, 768-hidden configuration:

```bash
python 4_pretrain.py \
  --outdir pretrain_1M_768D12H \
  --n_parts 100 \
  --hidden 768 \
  --batch_size 64 \
  --epochs 10 \
  --val_parts 5
```

Outputs:

```text
ckpt_pretrain/pretrain_1M_768D12H/encoder_pretrained.pt
ckpt_pretrain/pretrain_1M_768D12H/encoder_pretrained.best.pt
```

`4_pretrain.py` streams training shards, holds out fixed validation shards, and
saves the best encoder by validation loss. The root pretraining script uses the
default HSMP head count defined in `models/embedding.py`; the released checkpoint
bundle will provide the 1-, 3-, 6-, and 12-head pretrained variants.

## Predict Tg for PI1M or Acrylate/Acrylamide Libraries

`6_predict_PI1M.py` loads curated deployment checkpoints from:

```text
ckpt_deploy/<property>/split_<id>/finetuned.best.pt
```

For this release, the deployment checkpoints are Tg models over splits 0 through
9. They are distributed through the Google Drive checkpoint bundle, not committed
to GitHub.

### Predict a Generated Acrylate/Acrylamide Library
The paper applies Periodic-TDL to 48,208 systematically substituted vinyl
polymers from four acrylate- and acrylamide-based families: poly(acrylate),
poly(methacrylate), poly(acrylamide), and poly(methacrylamide). The same
inference path can be used for generated or literature libraries.

First, prepare:

```text
../data/processed/acrylate_v3_cleaned.csv
```

with a `smiles` column, then build periodic geometries and inference graphs:

```bash
python 1_periodic_geom.py --datafile acrylate_v3
python 2_build_complex_pretrain.py --datafile acrylate_v3 --procs 64 --chunks 5
```

Check `OUTDIR_DEFAULT` in `2_build_complex_pretrain.py` so the generated shards
are saved where you expect. Then run Tg inference over all deployment splits:

```bash
for i in {0..9}; do
  python 6_predict_PI1M.py \
    --deploy_datafile Tg \
    --out_csv acrylate_v3_Tg_split${i}.csv \
    --pretrain_hetero_dir data/HeteroGraphsAcrylate3 \
    --pretrain_stem acrylate_v3 \
    --split_id ${i}
done
```

Each output CSV contains:

```text
pSMILES,Tg_pred
```

## Notes

- Several paths are hard-coded in the scripts. Check `OUTDIR_DEFAULT`,
  `HETERO_DIR`, `PROCESSED_DIR`, `PT_CKPT_DIR`, and `VOCAB_DIR` before launching
  long jobs.
- The periodic Vietoris-Rips filtration uses fixed cutoffs at 2.0, 3.0, and
  4.0 Angstrom in the paper workflow.
- `1_periodic_geom.py` currently calls `build_periodic_dataset(..., n_jobs=64)`
  in its main block. Reduce this value if running on a smaller machine.
- `2_build_complex_finetune.py` expects both `smiles` and `value`; the
  pretraining/inference graph builder only requires `smiles`.
- Pretraining graph shards and downstream graph files are lists of PyG
  `HeteroData` objects. Downstream graphs carry `g.smiles` and `g.y`.
<!-- - `ARCHIVE/` is not part of the public workflow. It is kept for paper files,
  historical run commands, logs, and saved development outputs. -->
