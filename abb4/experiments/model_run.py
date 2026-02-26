import os
import wandb
import GPUtil
from omegaconf import DictConfig, OmegaConf
import numpy as np

import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from abb4.data.data_module import DataModule
from abb4.models.flow_module import FlowModule
from abb4.experiments import utils as eu

log = eu.get_pylogger(__name__) # multi-GPU-friendly python CLI logger
torch.set_float32_matmul_precision('high')

class ModelRun:

    def __init__(self, *, cfg: DictConfig, stage='train'):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment

        # initialise data module from config
        self._datamodule: LightningDataModule = DataModule(self._data_cfg)
        
        # initialize model from checkpoint or from scratch
        if stage == 'train': 
            if self._exp_cfg.warm_start_weights:
                self._model: LightningModule = self.load_model_from_checkpoint(self._exp_cfg.warm_start_weights,
                    map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())
                )
                log.info(f'Loaded model from {self._exp_cfg.warm_start_weights}')
            else:
                self._model: LightningModule = FlowModule(self._cfg)
        else:
            if stage == 'test':
                ckpt_dir = self._exp_cfg.warm_start
                self.ckpt_name = os.path.join(ckpt_dir, self._exp_cfg.testing.ckpt_name)
            elif stage == 'sample':
                self.ckpt_name = self._exp_cfg.prediction.ckpt_path
            
            self._model: LightningModule = self.load_model_from_checkpoint(self.ckpt_name,
                                                map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
            log.info(f'Loaded model from {self.ckpt_name}')
            self._model.eval()

    def load_model_from_checkpoint(self, checkpoint_path, map_location=None):
        return FlowModule.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            cfg=self._cfg,
            map_location=map_location,
            weights_only=False,
        )

    def setup_log_and_debug(self, stage=None):
        if self._exp_cfg.debug:
            log.info("Debug mode.")
            self.logger = None
            self._exp_cfg.num_devices = 1
            self._data_cfg.module.loaders.num_workers = 0
        elif stage == 'sample':
            self.logger = None
        else:
            self.logger = WandbLogger(**self._exp_cfg.wandb)

    def setup_trainer(self, mode=None):
        # get available devices
        # devices = GPUtil.getAvailable(order='memory', limit=self._exp_cfg.num_devices)
        devices = [int(x) for x in np.arange(self._exp_cfg.num_devices)]

        log.info(f"Using devices: {devices}")
        if len(devices) > 1:
            self._exp_cfg.trainer.accumulate_grad_batches = self._exp_cfg.trainer.accumulate_grad_batches//len(devices)
        
        callbacks = []
        if mode != 'sample':
            callbacks = [
                        ModelCheckpoint(**self._exp_cfg.checkpointer,
                                        filename='best-{epoch}-{step}'),
            ]

        trainer = Trainer(**self._exp_cfg.trainer,
                        # detect_anomaly=True,
                        num_sanity_val_steps=0,
                        callbacks=callbacks,
                        logger=self.logger,
                        # track_grad_norm=2,
                        use_distributed_sampler=False,  # using custom distributed batchsampler
                        enable_progress_bar=True,
                        enable_model_summary=True,
                        devices=devices
                        )
        
        # trainer.callbacks.append(
        #     ModelCheckpoint(
        #         monitor='valid/h_cdr3_W',
        #         mode='min',
        #         save_top_k=2,
        #         every_n_epochs=1,
        #         filename='best-H3-W-{epoch}'
        #     )
        # )

        # trainer.callbacks.append(
        #     ModelCheckpoint(
        #         monitor='valid/h_cdr3_recall',
        #         mode='min',
        #         save_top_k=2,
        #         every_n_epochs=1,
        #         filename='best-H3-recall-{epoch}'
        #     )
        # )

        # trainer.callbacks.append(
        #     ModelCheckpoint(
        #         monitor='valid/h_cdr1_d_mean_rmsd_sims',
        #         mode='min',
        #         save_top_k=2,
        #         every_n_epochs=1,
        #         filename='best-h_cdr1_d_mean_rmsd_sims-{epoch}-{step}'
        #     )
        # )

        # trainer.callbacks.append(
        #     ModelCheckpoint(
        #         monitor='valid/global_d_mean_rmsd_sims',
        #         mode='min',
        #         save_top_k=2,
        #         every_n_epochs=1,
        #         filename='best-global_d_mean_rmsd_sims-{epoch}-{step}'
        #     )
        # )

        return trainer

    def train_val(self):
        self.setup_log_and_debug()
 
        # set up checkpoint directory
        ckpt_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(ckpt_dir, exist_ok=True)
        log.info(f"Checkpoints and test samples will be saved to {ckpt_dir}.")
        
        # save train-val-test config to file and to wandb
        cfg_path = os.path.join(ckpt_dir, 'trvalte_config.yaml')
        with open(cfg_path, 'w') as f:
            OmegaConf.save(config=self._cfg, f=f.name)
        cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
        flat_cfg = dict(eu.flatten_dict(cfg_dict))
        if self.logger is not None: # don't do for debug
            if isinstance(self.logger.experiment.config, wandb.sdk.wandb_config.Config):
                self.logger.experiment.config.update(flat_cfg)

        trainer = self.setup_trainer(mode='train_val')

        if self._exp_cfg.load_all_states:
            ckpt_path = self._exp_cfg.warm_start_weights
        else:
            ckpt_path = None
        trainer.fit(
            model=self._model,
            datamodule=self._datamodule,
            ckpt_path=ckpt_path, # start new checkpointing after reloading model
        )

    def test(self):
        self.setup_log_and_debug()
        trainer = self.setup_trainer()
        trainer.test(model=self._model, datamodule=self._datamodule)
    
    def sample(self):
        self.setup_log_and_debug(stage='sample')

        # set-up directories to write samples and config to
        os.makedirs(self._exp_cfg.prediction.output_dir, exist_ok=True)
        config_path = os.path.join(self._exp_cfg.prediction.output_dir, 'sample_config.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._cfg, f=f)
        log.info(f'Saved config and samples to {self._exp_cfg.prediction.output_dir}')

        trainer = self.setup_trainer(mode='sample')
        trainer.predict(model=self._model, datamodule=self._datamodule)
