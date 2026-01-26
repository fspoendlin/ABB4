"""Utility functions for experiments."""
import logging, torch, os
import numpy as np
from abb4.analysis import utils as au
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf

def save_traj(
        sample: np.ndarray,
        bb_prot_traj: np.ndarray,
        x0_traj: np.ndarray,
        diffuse_mask: np.ndarray,
        output_dir: str,
        aa_traj = None,
        clean_aa_traj = None,
        write_trajectories = True,
    ):
    """Writes final sample and reverse diffusion trajectory.

    Args:
        bb_prot_traj: [noisy_T, N, 37, 3] atom37 sampled diffusion states.
            T is number of time steps. First time step is t=eps,
            i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
            N is number of residues.
        x0_traj: [clean_T, N, 37, 3] atom37 predictions of clean data at each time step.
        res_mask: [N] residue mask.
        diffuse_mask: [N] which residues are diffused.
        output_dir: where to save samples.
        aa_traj: [noisy_T, N] amino acids (0 - 20 inclusive).
        clean_aa_traj: [clean_T, N] amino acids (0 - 20 inclusive).
        write_trajectories: bool Whether to also write the trajectories as well
                                 as the final sample

    Returns:
        Dictionary with paths to saved samples.
            'sample_path': PDB file of final state of reverse trajectory.
            'traj_path': PDB file os all intermediate diffused states.
            'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
        b_factors are set to 100 for diffused residues
        residues if there are any.
    """

    diffuse_mask = diffuse_mask.astype(bool)
    noisy_traj_length, num_batch, num_res, _, _ = bb_prot_traj.shape

    for batch_item in range(num_batch):
        
        # use b-factors to specify which residues are diffused.
        this_diffuse_mask = diffuse_mask[batch_item, ...]
        b_factors = np.tile((this_diffuse_mask * 100)[:, None], (1, 37))

        this_prot_traj = bb_prot_traj[:, batch_item, :, :, :]
        this_x0_traj = x0_traj[:, batch_item, :, :, :]
        this_sample = sample[batch_item, :, :, :]
        this_aa_traj = aa_traj[:, batch_item, :] if aa_traj is not None else None
        this_clean_aa_traj = clean_aa_traj[:, batch_item, :] if clean_aa_traj is not None else None

        clean_traj_length = this_x0_traj.shape[0]
        assert this_sample.shape == (num_res, 37, 3)
        assert this_prot_traj.shape == (noisy_traj_length, num_res, 37, 3)
        assert this_x0_traj.shape == (clean_traj_length, num_res, 37, 3)
        if this_aa_traj is not None:
            assert this_aa_traj.shape == (noisy_traj_length, num_res)
            assert this_clean_aa_traj is not None
            assert this_clean_aa_traj.shape == (clean_traj_length, num_res)

        sample_path = os.path.join(output_dir[batch_item], 'sample.pdb')
        sample_path = au.write_prot_to_pdb(
            this_sample,
            sample_path,
            b_factors=b_factors,
            no_indexing=True,
            aatype=this_aa_traj[-1] if this_aa_traj is not None else None,
        )

        if write_trajectories:
            prot_traj_path = os.path.join(output_dir[batch_item], 'bb_traj.pdb') 
            prot_traj_path = au.write_prot_to_pdb(
                this_prot_traj,
                prot_traj_path,
                b_factors=b_factors,
                no_indexing=True,
                aatype=this_aa_traj,
            )
            x0_traj_path = os.path.join(output_dir[batch_item], 'x0_traj.pdb')
            x0_traj_path = au.write_prot_to_pdb(
                this_x0_traj,
                x0_traj_path,
                b_factors=b_factors,
                no_indexing=True,
                aatype=this_clean_aa_traj,
            )
    return

def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger

def flatten_dict(raw_dict):
    """
    Flattens a nested dict into a list of two- or three-tuples, 
    from which a flat dictionary can be reconstructed.
    """
    
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([
                (f'{k}:{i}', j) for i, j in flatten_dict(v)
            ])
        else:
            flattened.append((k, v))
    return flattened

def merge_configs(cfg: DictConfig, ckpt_new: DictConfig) -> DictConfig:
    
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(ckpt_new, False)
    cfg = OmegaConf.merge(cfg, ckpt_new)
    OmegaConf.set_struct(cfg, True)
    
    return cfg

def load_warmstart_config(cfg: DictConfig) -> DictConfig:
    log = get_pylogger(__name__) # multi-GPU-friendly python CLI logger
    
    if cfg.experiment.warm_start is not None and cfg.experiment.warm_start_cfg_override:
        # load warm-start config
        warm_start_cfg_path = os.path.join(
            os.path.dirname(cfg.experiment.warm_start), 'trvalte_config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        # save required values from current config
        warm_start_current = cfg.experiment.warm_start
        warm_start_weights_current = cfg.experiment.warm_start_weights
        load_all_states_current = cfg.experiment.load_all_states

        # over-write relevant fields with warm-start config
        cfg = merge_configs(cfg, warm_start_cfg)

        # set to original
        cfg.experiment['warm_start'] = warm_start_current
        cfg.experiment['warm_start_weights'] = warm_start_weights_current
        cfg.experiment['load_all_states'] = load_all_states_current
        log.info(f'Loaded warm start config from {warm_start_cfg_path}')
    
    return cfg

def fill_test_config(test_cfg: DictConfig) -> DictConfig:
    """Fills in missing fields in test config with values from training config."""
    log = get_pylogger(__name__) # multi-GPU-friendly python CLI logger

    warm_start_cfg_path = os.path.join(
            os.path.dirname(test_cfg.experiment.warm_start), 'trvalte_config.yaml')
    warm_start_cfg = OmegaConf.load(warm_start_cfg_path)
    # load warm-start config
    cfg = merge_configs(warm_start_cfg, test_cfg)
    log.info(f'Loaded config from {warm_start_cfg_path} and combined with test config')

    return cfg
