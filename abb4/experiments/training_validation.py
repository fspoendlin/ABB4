import hydra
import os
os.environ['HYDRA_FULL_ERROR'] = '1'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from model_run import *
import abb4.experiments.utils as eu

@hydra.main(version_base=None, config_path="../configs", config_name="training.yaml")
def train_val(cfg: DictConfig):
    cfg = eu.load_warmstart_config(cfg)
    run = ModelRun(cfg=cfg, stage='train')
    run.train_val()

if __name__ == "__main__":
    try:
        train_val()
    finally:
        if wandb.run is not None:
            wandb.finish()
        # should hopefully be called even upon SIGTERM
        # BUT wandb.finish() is known to sometimes be slow so SLURM's timeouts may be too short
        # which may send the node into drain
