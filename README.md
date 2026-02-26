# ABB4-STEROIDS: Antibody Conformational Ensemble Prediction

ABB4-STEROIDS is a generative structure prediction model for sampling **conformational ensembles** of antibodies rather than a single static structure. Conformational flexibility is central to antibody behavior, but exhaustive molecular dynamics (MD) is often expensive and many deep-learning methods focus on one structure at a time. This repository provides the ABB4-STEROIDS workflows for training, testing, and inference, built around large-scale simulation data (including coarse-grained and all-atom trajectories), with utilities to generate ensemble samples and post-process predictions into IMGT numbering.

## Table of Contents
- [Installation](#installation)
- [Repository Layout](#repository-layout)
- [Configuration](#configuration)
- [Running Inference](#running-inference)
- [Training Models](#training-models)
- [Testing / Evaluation](#testing--evaluation)
- [Examples / Usage](#examples--usage)
- [Scripts Overview](#scripts-overview)
- [License & Citation](#license--citation)

## Installation

### Requirements
- Linux environment (project is developed/run on Linux).
- Conda available on your system.
- Python 3.10 (used by `scripts/install.bash`).
- CUDA-capable GPUs are recommended for model training/inference (multi-GPU inference is supported via `torch.distributed.run`).

### Quick setup (recommended)
Run the provided installer script:

```bash
bash scripts/install.bash
```

This script creates a conda environment named `abb4_env`, installs PyTorch + dependencies, and installs the local package (`pip install .`).

### Notes
- If your shell does not auto-enable `conda activate` inside scripts, you may need to activate the environment manually before running project commands.
- Checkpoint paths and dataset CSV paths in config files are environment-specific and should usually be updated before running.

## Repository Layout

- `abb4/`: core package code.
	- `abb4/experiments/`: Hydra entrypoints and runtime orchestration.
	- `abb4/configs/`: YAML configs for training/testing/inference.
	- `abb4/data/`: data module, datasets, and batching.
	- `abb4/models/`: model + Lightning module implementation.
- `scripts/`: operational scripts (install, inference launcher, post-processing).
- `ckpt/`: checkpoints.
- `predictions/`: inference outputs.
- `outputs/`, `lightning_logs/`: run artifacts and logs.
- `metadata/`: metadata CSV files.
- `notebooks/`: example notebooks.

## Configuration

Configuration is managed with Hydra and YAML files in `abb4/configs/`.

### Key config files

| File | Purpose |
|------|---------|
| `abb4/configs/training.yaml` | Training defaults composition |
| `abb4/configs/testing.yaml` | Test-time overrides |
| `abb4/configs/inference.yaml` | Inference/prediction overrides |
| `abb4/configs/experiment.yaml` | Trainer, optimizer, scheduler, logging, checkpointing |
| `abb4/configs/data.yaml` | Dataset paths, data splits, dataloader settings |
| `abb4/configs/model.yaml` | Model architecture settings |
| `abb4/configs/interpolant.yaml` | Sampling/interpolant settings |

### What to edit most often
- **Data locations**: `data.dataset.*.csv_path` in `data.yaml` and `inference.yaml`.
- **Checkpoint paths**:
	- training warm-start: `experiment.warm_start` / `experiment.warm_start_weights`
	- inference checkpoint: `experiment.prediction.ckpt_path`
- **Inference output path**: `experiment.prediction.output_dir` in `inference.yaml`.
- **Batching/performance**:
	- `data.module.loaders.num_workers`
	- `data.dataset.predict.pred_sampler.batch_size`
	- `experiment.num_devices`

### Example: adjust inference inputs/outputs
Edit `abb4/configs/inference.yaml`:
- `data.dataset.predict.csv_path`: input CSV with prediction targets
- `data.dataset.predict.num_samples`: number of generated samples per input
- `experiment.prediction.ckpt_path`: model checkpoint
- `experiment.prediction.output_dir`: prediction directory

## Running Inference

### Canonical command

```bash
bash scripts/inference.bash --nproc_per_node 4 --master_port 29500
```

### Script flags
- `--nproc_per_node` (default: `1`)
- `--master_port` (default: `29500`)

### Under the hood
The script launches:

```bash
python -W ignore -m torch.distributed.run \
	--nproc_per_node=<N> \
	--master_port=<PORT> \
	abb4/experiments/inference.py -cn inference
```

### Inputs
- Inference config: `abb4/configs/inference.yaml`
- Input CSV: `data.dataset.predict.csv_path`
- Checkpoint: `experiment.prediction.ckpt_path`

### Outputs
- Config snapshot: `<output_dir>/sample_config.yaml`
- Per target directory output files:
	- `<output_dir>/<pdb_name>/sample_<copy_index>.pdb`
	- optionally (if enabled in config):
		- `sample_<copy_index>_clean_traj.pdb`
		- `sample_<copy_index>_int_traj.pdb`

### Post-processing
To convert predicted PDBs to IMGT numbering:

```bash
python scripts/renumber_predictions.py --pred_path predictions --out_path predictions_imgt --gpu
```

Arguments:
- `--pred_path` (required)
- `--out_path` (optional, defaults to `<pred_path>_imgt` behavior)
- `--gpu` (optional flag)

## Training Models

### Canonical command

```bash
python abb4/experiments/training_validation.py
```

This entrypoint uses Hydra config `training.yaml`, initializes `ModelRun(stage='train')`, and calls `Trainer.fit(...)`.

### Training flow (high level)
1. Parse config (`training.yaml` + defaults).
2. Optionally merge warm-start config (`trvalte_config.yaml`) when enabled.
3. Build `DataModule(cfg.data)` and `FlowModule(cfg)`.
4. Configure optimizer/scheduler from `experiment.*` settings.
5. Run Lightning training loop and validation.
6. Save checkpoints and config snapshot.

### Checkpoints and outputs
- Checkpoint directory: `experiment.checkpointer.dirpath`
- Training config snapshot: `<checkpointer.dirpath>/trvalte_config.yaml`
- Logs/artifacts: `outputs/`, `lightning_logs/` (and W&B if enabled)

### About training stages
The project/paper reports multi-stage model development; operationally, this repo exposes a single training entrypoint with warm-start support, which is how staged training is typically chained in practice.

## Testing / Evaluation

Run test-time evaluation with:

```bash
python abb4/experiments/testing.py
```

This uses `abb4/configs/testing.yaml`, fills missing test fields from the training checkpoint config, and runs `Trainer.test(...)` through `ModelRun(stage='test')`.

## Examples / Usage

- Explore the notebook examples in `notebooks/` (for example, `notebooks/test.ipynb`).
- For command-line workflows, see inference/training/testing examples above.

## Scripts Overview

Canonical tracked scripts in `scripts/`:

| Script | Purpose | Example |
|--------|---------|---------|
| `scripts/install.bash` | Environment and dependency setup | `bash scripts/install.bash` |
| `scripts/inference.bash` | Distributed inference launcher | `bash scripts/inference.bash --nproc_per_node 4 --master_port 29500` |
| `scripts/renumber_predictions.py` | IMGT renumbering for predicted PDBs | `python scripts/renumber_predictions.py --pred_path predictions --out_path predictions_imgt --gpu` |

## License & Citation

### License
This repository includes a `LICENSE` file at the project root. See `LICENSE` for full terms.

### Citation
If you use ABB4-STEROIDS in your work:
1. Cite the ABB4-STEROIDS method/manuscript associated with this repository.
2. Cite this GitHub repository (URL + access date).

If a formal citation file or DOI is added later, prefer that canonical citation format.

