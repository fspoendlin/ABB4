import hydra

from model_run import *
import abb4.experiments.utils as eu

@hydra.main(version_base=None, config_path="../configs", config_name="testing.yaml")
def test(cfg: DictConfig) -> None:
    cfg = eu.fill_test_config(cfg)
    run = ModelRun(cfg=cfg, stage='test')
    run.test()
    
if __name__ == "__main__":
    try:
        test()
    finally:
        wandb.finish()