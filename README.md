# ABB4-STEROIDS: Antibody Conformational Ensemble Prediction

ABB4-STEROIDS is a generative structure prediction model for sampling **conformational ensembles** of antibodies rather than a single static structure. Conformational flexibility is central to antibody behavior, but exhaustive molecular dynamics (MD) is often expensive and many deep-learning methods focus on one structure at a time. This repository provides workflows for training, testing, and inference, built around large-scale simulation data (coarse-grained and all-atom trajectories), using flow matching on SE(3) to generate diverse antibody conformations.

## Table of Contents
- [Installation](#installation)
- [Running Inference](#running-inference)
- [Analysis](#post-processing--analysis)
- [Training Models](#training-models)
- [Configuration Reference](#configuration-reference)
- [Repository Layout](#repository-layout)
---

## Installation

**Requirements**: Conda, Python 3.10, CUDA >= 12.6.

```bash
bash scripts/install.bash
```

This creates a conda environment `abb4_env`, installs PyTorch 2.8.0 + all dependencies, and installs the local package.

The script automatically detects your installed CUDA version via `nvidia-smi` and selects the best matching wheel from the following supported tags (in descending preference):

| CUDA version | Wheel tag |
|---|---|
| 12.9 | `cu129` |
| 12.8 | `cu128` |
| 12.6 | `cu126` |

If your CUDA version falls between two entries, the next lower tag is used (e.g. CUDA 12.7 → `cu126`). The script will print the selected tag and ask for confirmation before installing.

**If installation fails**, you can override the CUDA tag manually by editing the two pip lines in `scripts/install.bash` directly:
```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/<CUDA_TAG>
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+<CUDA_TAG>.html
```
replacing `<CUDA_TAG>` with the appropriate tag from the table above.

---

## Running Inference

### 1. Create an input CSV

Create a CSV file with one row per antibody. The following columns are required:

| Column | Description |
|--------|-------------|
| `pdb_name` | Name/identifier for the query antibody |
| `VH_seq` | VH (heavy chain) sequence |
| `VL_seq` | VL (light chain) sequence |

Example:

```csv
pdb_name,VH_seq,VL_seq
my_antibody,EVQLVESGGGLVQPGGSLRLSCAAS...,DIQMTQSPSSLSASVGDRVTITC...
```

### 2. Edit the inference config

Open `abb4/configs/inference.yaml` and set the following key parameters:

```yaml
data:
  module:
    loaders:
      num_workers: 10           # CPU workers for data loading; set to number of available CPUs if lower

  dataset:
    predict:
      csv_path: /path/to/input.csv   # path to your input CSV (created above)
      num_samples: 100               # number of structure to sample per antibody
      pred_sampler:
        batch_size: 100              # recommended for A100 (80GB), reduce if required

interpolant:
  sampling:
    num_timesteps: 100               # ODE steps during inference
                                     # 100 matches publication accuracy; 50 gives idenitcal accuracy with ~2x speedup

experiment:
  num_devices: 1                     # number of GPUs to use
  prediction:
    ckpt_path: ckpt/abb4_STEROIDS/abb4_STEROIDS.ckpt  # which model to use (see options below)
    output_dir: /path/to/output/dir  # where generated PDB files are written
```

**Available checkpoints** (`ckpt_path`):

| Checkpoint | Description |
|------------|-------------|
| `ckpt/abb4_STEROIDS/abb4_STEROIDS.ckpt` | **(default)** Full model trained on CG + all-atom MD — best for conformational ensemble sampling |
| `ckpt/abb4_base/abb4_base.ckpt` | Trained on experimental structures only — use for single-structure prediction |
| `ckpt/abb4_STEROIDS_CG/abb4_STEROIDS_CG.ckpt` | Trained on CG MD only (no all-atom fine-tuning) — conformational sampling without all-atom refinement |

### 3. Run inference

Set `--nproc_per_node` to match `experiment.num_devices` in the config.

```bash
# Multi-GPU
bash scripts/inference.bash --nproc_per_node 4 --master_port 29500

# Single GPU
bash scripts/inference.bash --nproc_per_node 1
```

### 4. Outputs

```
<output_dir>/
  sample_config.yaml                       # config snapshot for reproducibility
  <target_name>/
    sample_0.pdb                           
    ...
    sample_100.pdb
```

### 5. Post-processing: IMGT renumbering (recommended)

Convert predicted PDBs to standard IMGT antibody numbering:

```bash
python scripts/renumber_predictions.py \
  --pred_path /path/to/output/dir \
  --out_path /path/to/output/dir_imgt \
  --gpu
```


## Analysis

Scripts are provided to perform preliminary analysis of the predicted ensembles. RMSF and RMSD values can be calculated:

```bash
# Calculate CDR RMSD statistics across ensemble
python scripts/calculate_cdr_rmsds.py \
  --pred_path /path/to/output/dir_imgt \
  --n_jobs 20
# Output: /path/to/output/dir_imgt/cdr_rmsds.csv

# Calculate per-residue RMSF (flexibility) across ensemble
python scripts/calculate_cdr_rmsfs.py \
  --pred_path /path/to/output/dir_imgt
# Output: /path/to/output/dir_imgt/cdr_rmsfs.csv
```

---
---

## Training Models

### 1. Edit the training configs

**Data** — `abb4/configs/data.yaml`:
```yaml
data:
  dataset:
    train_val_test_pdbs:
      csv_path: /path/to/metadata.csv     # training/validation/test metadata
```

**Warm-start (optional)** — `abb4/configs/experiment.yaml`:
```yaml
experiment:
  warm_start: /path/to/checkpoint/dir     # directory containing trvalte_config.yaml
  warm_start_weights: /path/to/checkpoint/dir/epoch=N.ckpt
  warm_start_cfg_override: true
```

**Checkpoints & logging** — `abb4/configs/experiment.yaml`:
```yaml
experiment:
  checkpointer:
    dirpath: /path/to/save/checkpoints
  logger:
    use: wandb                            # set to null to disable W&B
    project: ab_folding
    group: ensemble
```

### 2. Run training

```bash
python abb4/experiments/training_validation.py
```

Override any config value on the command line:
```bash
python abb4/experiments/training_validation.py \
  data.dataset.train_val_test_pdbs.csv_path=/new/data.csv \
  experiment.num_devices=2
```

### 3. Key training parameters

| Parameter | Config file | Key | Default | Effect |
|-----------|------------|-----|---------|--------|
| Training data CSV | `data.yaml` | `data.dataset.train_val_test_pdbs.csv_path` | — | Training/val/test metadata |
| Number of GPUs | `experiment.yaml` | `experiment.num_devices` | 4 | GPUs for DDP training |
| Learning rate | `experiment.yaml` | `experiment.optimizer.lr` | 1e-6 | Main LR |
| Weight decay | `experiment.yaml` | `experiment.optimizer.weight_decay` | 1e-2 | AdamW regularisation |
| LR scheduler | `experiment.yaml` | `experiment.lr_scheduler.use` | `warmup_cosine_annealing` | Scheduler type |
| Max epochs | `experiment.yaml` | `experiment.trainer.max_epochs` | 5000 | Training budget |
| Gradient accumulation | `experiment.yaml` | `experiment.trainer.accumulate_grad_batches` | 5 | Effective batch multiplier |
| Train batch size | `data.yaml` | `data.module.loaders.batch_size` | 10 | Per-GPU, single structures |
| Val ensemble size | `data.yaml` | `data.dataset.val.ensemble_size` | 50 | Structures stacked for val |
| Checkpoint metric | `experiment.yaml` | `experiment.checkpointer.monitor` | `valid/h_cdr3_d_mean_rmsd_sims` | Saved on min CDR H3 RMSD |
| Translation loss weight | `experiment.yaml` | `experiment.training.training_losses.translation_loss_weight` | 8.0 | SE(3) translation VF |
| Rotation loss weight | `experiment.yaml` | `experiment.training.training_losses.rotation_loss_weight` | 0.8 | SO(3) rotation VF |
| FAPE loss weight | `experiment.yaml` | `experiment.training.training_losses.A_fape_loss_weight` | 0.08 | All-atom FAPE |
| Torsion loss weight | `experiment.yaml` | `experiment.training.training_losses.torsion_loss_weight` | 0.4 | Backbone/side-chain torsions |
| RMSF loss weight | `experiment.yaml` | `experiment.training.training_losses.rmsf_loss_weight` | 0.05 | Ensemble flexibility |
| Optimal transport | `interpolant.yaml` | `interpolant.ot.use_ot` | true | OT noise matching |
| DataLoader workers | `data.yaml` | `data.module.loaders.num_workers` | 10 | Increase for fast storage |

### 4. Outputs

```
<checkpointer.dirpath>/
  epoch=N-step=M.ckpt        # model checkpoints (top-5 by CDR H3 RMSD)
  last.ckpt                  # always-updated latest checkpoint
  trvalte_config.yaml        # full config snapshot (used for warm-start)
outputs/                     # Hydra run artifacts
lightning_logs/              # Lightning progress logs
wandb/                       # W&B run data (if enabled)
```

---

## Configuration Reference

All configs are in `abb4/configs/` and use [Hydra](https://hydra.cc/). The file `training.yaml` composes the others as defaults.

| File | Purpose |
|------|---------|
| `training.yaml` | Top-level training composition |
| `data.yaml` | Dataset paths, splits, batching, OT |
| `model.yaml` | Architecture: embedding dims, IPA blocks, heads, dropout |
| `interpolant.yaml` | Flow matching: noise schedule, OT distance, sampling steps |
| `experiment.yaml` | Optimizer, scheduler, loss weights, trainer, logging, checkpointing |
| `inference.yaml` | Inference overrides: checkpoint, input/output paths, num_samples |
| `testing.yaml` | Test-time overrides: checkpoint dir, test CSV |

**Hydra CLI syntax** — override any dotted config key:
```bash
python abb4/experiments/training_validation.py key1=value1 key2=value2
```

---


## Repository Layout

```
ABB4/
├── abb4/
│   ├── configs/        # Hydra YAML configs
│   ├── data/           # DataModule, datasets, Interpolant, preprocessing
│   ├── experiments/    # Entrypoints + ModelRun orchestrator
│   ├── models/         # FlowModel, FlowModule (Lightning), losses
│   └── analysis/       # PDB I/O utilities
├── openfold/           # Bundled OpenFold (IPA, rigid body ops)
├── scripts/            # Operational scripts
├── ckpt/               # Model checkpoints
├── predictions/        # Inference outputs
├── data/               # Dataset CSVs
└── notebooks/          # Example notebooks
```

---

## License & Citation

See `LICENSE` for full terms.

If you use ABB4-STEROIDS in your work, please cite the associated manuscript and this repository.
