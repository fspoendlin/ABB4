#!/bin/bash

conda create -n abb4_env python=3.10
conda activate abb4_env
conda install pytorch=2.8.0 pytorch_scatter -c conda-forge
conda install -c conda-forge biopython matplotlib numpy pandas pyyaml scipy seaborn dm-tree tqdm wandb
pip install mdtraj pytorch-lightning hydra-core GPUtil ml-collections lightning anarcii POT
pip install .
