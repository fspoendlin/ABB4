#!/bin/bash

source ~/.bashrc
cd /ceph/opig-shared/users/haque/DPhil/ABB4/ABB4
conda create -n abb4_env python=3.10
conda activate abb4_env
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
conda install -c conda-forge biopython matplotlib numpy pandas pyyaml scipy seaborn dm-tree tqdm wandb
pip install mdtraj pytorch-lightning hydra-core GPUtil ml-collections lightning anarcii POT
pip install .
