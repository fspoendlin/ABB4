import torch
import copy
import math
import ot as pot
from ot.backend import get_backend
import pandas as pd
import functools as fn
import torch.nn.functional as F
from abb4.data import so3_utils, all_atom, OT_utils
from abb4.data import utils as du
from scipy.spatial.transform import Rotation


def _centered_gaussian(num_batch, num_res, device):
    # (num_res * xyz) gaussian noise for each protein
    # centered at exactly (0,0,0) per protein
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def _uniform_so3(num_batch, num_res, device):
    # uniformly randomly sampled rotation *matrices* in SO(3)
    # (one per residue per protein)
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

def _all_mask_token(num_batch, num_res, device):
    return torch.ones(
        num_batch, num_res, device=device) * du.MASK_TOKEN_INDEX

def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])

def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (    rotmats_t * diffuse_mask[..., None, None]
                + rotmats_1 * (1 - diffuse_mask[..., None, None])   )

def _aatypes_diffuse_mask(aatypes_t, aatypes_1, diffuse_mask):
    return aatypes_t * diffuse_mask + aatypes_1 * (1 - diffuse_mask)

def _logits_diffuse_mask(logits_t, logits_1, diffuse_mask):
    return _trans_diffuse_mask(logits_t, logits_1, diffuse_mask)

def _structure_noise(trans_1, rotmats_1, res_mask, igso3, rots_dist, device):
    # trans noise
    trans_0 = _centered_gaussian(*res_mask.shape, device)
    trans_0 = trans_0 * du.NM_TO_ANG_SCALE

    trans_0, _, _ = du.batch_align_structures(
        trans_0, trans_1, mask=res_mask
    )

    num_batch, num_res = res_mask.shape
    if rots_dist == 'USO3':
        rotmats_0 = _uniform_so3(num_batch, num_res, device)
    elif rots_dist == 'IGSO3':
        # rotmats noise
        rotmats_noisy = igso3.sample(
            torch.tensor([1.5]),
            num_batch*num_res
        )
        rotmats_noisy = rotmats_noisy.reshape(num_batch, num_res, 3, 3).to(device)
        rotmats_0 = torch.einsum(
            # difference between complete noise (t=0) and correct matrix (t=1)
            "...ij,...jk->...ik", rotmats_1, rotmats_noisy)
    else:
        raise ValueError(f'Unknown rots_dist {rots_dist}')

    return trans_0, rotmats_0

def _subset_rigids(subset, device):
    subset_trans_1 = []
    subset_rotmats_1 = []

    for processed_file_path in subset.processed_path.values:
        trans, rotmat = du.load_rigids(processed_file_path)
        subset_trans_1.append(trans)
        subset_rotmats_1.append(rotmat)

    try:
        subset_trans_1 = torch.stack(subset_trans_1).to(device)
        subset_rotmats_1 = torch.stack(subset_rotmats_1).to(device)
    except RuntimeError:
        raise ValueError('Datapoints with different number of residues encountered')

    return subset_trans_1, subset_rotmats_1

def _dataset_OT(batch, set_csv, ot_fn, ot_cfg, igso3, rots_dist, device):
    num_batch, num_res = batch['res_mask'].shape
    trans_0 = torch.zeros(num_batch, num_res, 3, device=device)
    rotmats_0 = torch.zeros(num_batch, num_res, 3, 3, device=device)

    for i, pdb_name_enc in enumerate(batch['pdb_name_enc']):
        target = set_csv.query(f'pdb_name_enc == {pdb_name_enc.item()}')
        assert target.shape[0] == 1 # there should be exactly one target per encoding
        # subset of data with identical value in specified column
        col = ot_cfg.ot_dataset_col
        subset = set_csv[(set_csv[col] == target[col].item())] # & 
                        #  (set_csv.pdb_name_enc != pdb_name_enc.item())]
        # n_samples = min(subset.shape[0], ot_cfg.ot_max_samples)

        if subset.shape[0]: # no OT if no other structures in subset
            i_trans_0, i_rotmats_0 = _structure_noise(
                batch['trans_1'][i][None,...],
                batch['rotmats_1'][i][None,...],
                batch['res_mask'][i][None,...],
                igso3,
                rots_dist,
                device
            )

            trans_0[i] = i_trans_0.squeeze()
            rotmats_0[i] = i_rotmats_0.squeeze()
            continue
        
        # subset = subset.sample(n_samples, replace=False, random_state=0)
        subset = subset.sample(ot_cfg.ot_max_samples, replace=True, weights='joint_proba_norm')
        subset = pd.concat([target, subset])

        subset_trans_1, subset_rotmats_1 = _subset_rigids(subset, device)
        num_subset, num_subs_res = subset_trans_1.shape[:2]
        subset_mask = batch['res_mask'][i][:num_subs_res].unsqueeze(0)
        subset_mask = subset_mask.repeat(num_subset, 1)

        # trans noise
        subset_trans_0 = _centered_gaussian(*subset_trans_1.shape[:2], device) 
        subset_trans_0 *= du.NM_TO_ANG_SCALE

        if rots_dist == 'USO3':
            subset_rotmats_0 = _uniform_so3(num_subset, num_subs_res, device)
        elif rots_dist == 'IGSO3':
            # rotmats noise
            subset_noisy_rotmats = igso3.sample(
                    torch.tensor([1.5]),
                    num_subset*num_subs_res
                ).reshape(num_subset, num_subs_res, 3, 3).to(device)
            subset_rotmats_0 = torch.einsum(
                "...ij,...jk->...ik", subset_rotmats_1, subset_noisy_rotmats)
        else:
            raise ValueError(f'Unknown rots_dist {rots_dist}')

        # cost martices
        r3_dist, subset_trans_0_aligned = OT_utils.batch_r3_dist_matrix(
            subset_trans_0, subset_trans_1, subset_mask, num_subset, num_subs_res, return_aligned_noise=True
            )
        so3_dist = OT_utils.batch_so3_dist_matrix(
            subset_rotmats_0, subset_rotmats_1, subset_mask, num_subset, num_subs_res
            )

        se3_dist = ot_cfg.so3_weight * torch.abs(so3_dist) + torch.abs(r3_dist)

        if ot_cfg.ot_distance == 'se3':
            cost = se3_dist
        elif ot_cfg.ot_distance == 'so3':
            cost = so3_dist
        elif ot_cfg.ot_distance == 'r3':
            cost = r3_dist
        
        # uniform_dist = pot.unif(num_subset, type_as=cost) # throws cuda error when OT run in dataset, cause the tensor is incorrectly initialized on cuda not cpu
        uniform_dist = torch.ones(num_subset, device=device) / num_subset

        if device == torch.device('cpu'): # POT throws an error when tensor on cpu but cuda available
            cost = cost.numpy()
            uniform_dist = uniform_dist.numpy()

        T = ot_fn(
            uniform_dist, # uniform dist over datapoint
            uniform_dist,
            cost
        )

        if device == torch.device('cpu'):
            T = torch.tensor(T, device=device)

        T_target = T[:,0].squeeze() # OT all noise -> target structure (index 0)
        T_target /= torch.sum(T_target)
        idx_source = torch.multinomial(T_target, 1) # index of noise

        trans_0[i, :num_subs_res, :] = subset_trans_0_aligned[idx_source,0].squeeze() # i: noise, j=0: target
        rotmats_0[i, :num_subs_res, :] = subset_rotmats_0[idx_source].squeeze() # i: noise, no j
    
    return trans_0, rotmats_0

class Interpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        self.task = cfg.task
        self.gen_extent = cfg.gen_extent
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._aatypes_cfg = cfg.aatypes
        self._sample_cfg = cfg.sampling
        self.num_tokens = 21 if self._aatypes_cfg.interpolant_type == "masking" else 20
        self._igso3 = None

        self.csv = None
        self.train_csv = None
        self.val_csv = None
        self.test_csv = None
        self.extra_val_csv = None
        self.extra_test_csv = None

        should_load_dataset_csv = cfg.get('load_dataset_csv', True)
        if should_load_dataset_csv:
            if cfg.csv_path is None:
                raise ValueError('interpolant.csv_path must be set when interpolant.load_dataset_csv=True')
            if cfg.max_resolution is None:
                raise ValueError('interpolant.max_resolution must be set when interpolant.load_dataset_csv=True')

            csv = pd.read_csv(cfg.csv_path)
            self.csv = csv[csv.Resolution <= cfg.max_resolution]
            self.train_csv = csv[csv.split == 'train']
            self.val_csv = csv[csv.split == 'val']
            self.test_csv = csv[csv.split == 'test']
            self.extra_val_csv = self.csv[self.csv.split == 'val_extra']
            self.extra_test_csv = self.csv[self.csv.split == 'test_extra']

        if self._cfg.ot.ot_fn == "emd":
            self.ot_fn = pot.emd
        else:
            raise ValueError(f'OT function {self._cfg.ot.ot_fn} not supported yet')
        
    @property
    def igso3(self):
        # isotropic Gaussian-equivalent on SO3 
        # (=truncated-series form of closed-form solution to Wiener diffusion process on SO3)
        # see FrameDiff paper, Appendix E
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch, shared_t=False, zero_t=False):
        # in (0,1) - see Multiflow paper App. B.2, last paragraph
        if zero_t: # sample only time step 0, train a single step
            t = torch.zeros(num_batch, device=self._device)
        elif shared_t: # same t for entire batch
            t = torch.rand(1, device=self._device).repeat(num_batch)
        else:
            t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2*self._cfg.min_t) + self._cfg.min_t

    def _batch_OT(self, batch, rots_dist):
        num_batch, num_res = batch['res_mask'].shape
        # trans noise
        batch_trans_0 = _centered_gaussian(*batch['res_mask'].shape, self._device)
        batch_trans_0 *= du.NM_TO_ANG_SCALE

        if rots_dist == 'USO3':
            batch_rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        elif rots_dist == 'IGSO3':
            # rotmats noise
            batch_noisy_rotmats = self.igso3.sample(\
                    torch.tensor([1.5]),
                    num_batch*num_res
                ).reshape(num_batch, num_res, 3, 3).to(self._device)
            batch_rotmats_0 = torch.einsum(
                "...ij,...jk->...ik", batch['rotmats_1'], batch_noisy_rotmats)
        else:
            raise ValueError(f'Unknown rots_dist {rots_dist}')

        r3_dist, batch_trans_0_aligned = OT_utils.batch_r3_dist_matrix(
            batch_trans_0, batch['trans_1'], batch['res_mask'], num_batch, num_res, return_aligned_noise=True
        )
        so3_dist = OT_utils.batch_so3_dist_matrix(
            batch_rotmats_0, batch['rotmats_1'], batch['res_mask'], num_batch, num_res
        )
        se3_dist = self._cfg.ot.so3_weight * torch.abs(so3_dist) + torch.abs(r3_dist)

        if self._cfg.ot.ot_distance == 'se3':
            cost = se3_dist
        elif self._cfg.ot.ot_distance == 'so3':
            cost = so3_dist
        elif self._cfg.ot.ot_distance == 'r3':
            cost = r3_dist

        T = self.ot_fn(
            pot.unif(num_batch, type_as=se3_dist), # uniform dist over datapoint
            pot.unif(num_batch, type_as=se3_dist),
            cost
        )

        # Noise permutation
        # T[i,j] is the probability of matching noise i to ground truth j
        noise_perm = torch.multinomial(T.T, 1).squeeze()

        OT_trans_0 = batch_trans_0_aligned[noise_perm].diagonal(dim1=0, dim2=1).permute(2,0,1)
        assert OT_trans_0.shape == (num_batch, num_res, 3)
        OT_rotmats_0 = batch_rotmats_0[noise_perm]
        assert OT_rotmats_0.shape == (num_batch, num_res, 3, 3)
  
        return OT_trans_0, OT_rotmats_0
    
    def _corrupt_structure(self, batch, stage=None):

        if not self._cfg.ot.use_ot:
            trans_0, rotmats_0 = _structure_noise(
                batch['trans_1'], batch['rotmats_1'], batch['res_mask'], 
                self.igso3, self._cfg.rots.dist, self._device
            )

        else: # Optimal transport
            if self._cfg.ot.ot_mode == 'dataset':
                if 'trans_0' in batch: # precomputed in dataset
                    trans_0 = batch['trans_0']
                    rotmats_0 = batch['rotmats_0']
                else:
                    assert stage is not None , \
                    "stage must be provided when using OT across dataset mode"
                    if stage == 'train':
                        set_csv = self.train_csv
                    elif stage == 'valid':
                        set_csv = self.val_csv
                    elif stage == 'extra_valid':
                        set_csv = self.extra_val_csv
                    elif stage == 'test':
                        set_csv = self.test_csv
                    elif stage == 'extra_test':
                        set_csv = self.extra_test_csv
                    else:
                        raise ValueError(f'Unknown stage {stage} for OT dataset mode')
                    assert set_csv.shape[0] > 0, f'{stage} split not found in CSV'

                    trans_0, rotmats_0 = _dataset_OT(
                        batch, set_csv, self.ot_fn, self._cfg.ot, self.igso3, 
                        self._cfg.rots.dist, self._device
                        )
            elif self._cfg.ot.ot_mode == 'batch':
                trans_0, rotmats_0 = self._batch_OT(batch, self._cfg.rots.dist)
            else:
                raise ValueError(f'OT mode {self._cfg.ot.ot_mode} not supported')

        # interpolate 0 to t
        res_mask = batch['res_mask']
        diffuse_mask = batch['diffuse_mask']

        trans_t = ((1 - batch['t_trans'][..., None]) * trans_0 + 
                   batch['t_trans'][..., None] * batch['trans_1'])
        trans_t = _trans_diffuse_mask(trans_t, batch['trans_1'], diffuse_mask)
        trans_t = trans_t * res_mask[..., None]

        rotmats_t = so3_utils.geodesic_t(
            batch['t_rot'][..., None], batch['rotmats_1'], rotmats_0
        )
        identity = torch.eye(3, device=self._device)
        rotmats_t = (
            rotmats_t * res_mask[..., None, None]
            + identity[None, None] * (1 - res_mask[..., None, None])
        )
        rotmats_t = _rots_diffuse_mask(rotmats_t, batch['rotmats_1'], diffuse_mask)

        return trans_t, rotmats_t

    def _corrupt_aatypes(self, aatypes_1, t, res_mask, diffuse_mask):
        num_batch, num_res = res_mask.shape
        assert aatypes_1.shape == (num_batch, num_res)
        assert res_mask.shape == (num_batch, num_res)
        assert diffuse_mask.shape == (num_batch, num_res)
        assert t.shape == (num_batch, 1)
        
        u = torch.rand(num_batch, num_res, device=self._device)
        aatypes_t = aatypes_1.clone()
        corruption_mask = u < (1-t) # (num_batch, num_res)

        if self._aatypes_cfg.interpolant_type == "masking":
            aatypes_t[corruption_mask] = du.MASK_TOKEN_INDEX
        elif self._aatypes_cfg.interpolant_type == "uniform":
            uniform_sample = torch.randint_like(aatypes_t, low=0, high=du.NUM_TOKENS)
            aatypes_t[corruption_mask] = uniform_sample[corruption_mask]
        else:
            raise ValueError(f"Unknown aatypes interpolant type {self._aatypes_cfg.interpolant_type}")
        
        # mask anywhere outside the considered area
        aatypes_t = aatypes_t * res_mask + du.MASK_TOKEN_INDEX * (1 - res_mask)
        # only corrupt in area to be generated
        return _aatypes_diffuse_mask(aatypes_t, aatypes_1, diffuse_mask)

    def corrupt_batch(self, batch, stage=None):
        """Called during training only."""
        noisy_batch = copy.deepcopy(batch)
        # [B, N, 3]
        trans_1 = batch['trans_1']  # Angstrom
        # [B, N, 3, 3]
        rotmats_1 = batch['rotmats_1']
        # [B, N]
        aatypes_1 = batch['aatypes_1']

        # [B, N]
        res_mask = batch['res_mask']            # area to be considered
        diffuse_mask = batch['diffuse_mask']    # area to be generated
        num_batch, num_res = diffuse_mask.shape

        # [B, 1]
        if self._cfg.separate_t:
            """ From Multiflow paper, App. J2:
            'When training with our t, t' objective that enables the model to learn over different relative levels of corruption between 
            structure and sequence, 10% of the time we set t = 1 and draw t' ∼ U(0, 1) and 10% of the time we set t' = 1 and draw t ∼ U(0, 1). 
            The remaining 80% of the time we draw both t and t' independently from t, t' ∼ U(0, 1).'
            """
            # select samples to co-gen/fwd/inv fold
            u = torch.rand((num_batch,), device=self._device)
            forward_fold_mask = (u < self._cfg.fwd_fold_prob).float()
            inverse_fold_mask = (u < self._cfg.fwd_fold_prob + self._cfg.inv_fold_prob).float() * \
                (u >= self._cfg.fwd_fold_prob).float()

            # sample t, t'
            tau_cogen_struc = self.sample_t(num_batch, shared_t=self._cfg.shared_t, zero_t=self._cfg.zero_t)
            tau_cogen_seq = self.sample_t(num_batch, shared_t=self._cfg.shared_t, zero_t=self._cfg.zero_t)
            t_cogen_struc_trans = self.sample_kappa(tau_cogen_struc, component='trans')
            t_cogen_struc_rots = self.sample_kappa(tau_cogen_struc, component='rots')
            t_cogen_seq =  self.sample_kappa(tau_cogen_seq, component='seq')
            all_ones = torch.ones((num_batch,), device=self._device)

            # t_seq should be 1 for fwd folding, uniform otherwise
            t_seq = forward_fold_mask * all_ones + (1 - forward_fold_mask) * t_cogen_seq
            # struc_ts should be 1 for inv folding, uniform otherwise
            t_struc_trans = inverse_fold_mask * all_ones + (1 - inverse_fold_mask) * t_cogen_struc_trans
            t_struc_rots = inverse_fold_mask * all_ones + (1 - inverse_fold_mask) * t_cogen_struc_rots

            t_trans = t_struc_trans[:, None]
            t_rot = t_struc_rots[:, None]
            t_seq = t_seq[:, None]
        else:
            tau = self.sample_t(num_batch, shared_t=self._cfg.shared_t, zero_t=self._cfg.zero_t)[:, None] # [B, 1]
            t_trans = self.sample_kappa(tau, component='trans')
            t_rot = self.sample_kappa(tau, component='rots')
            t_seq = self.sample_kappa(tau, component='seq')
            
        # apply corruptions
        if self.task == 'cogen' or self.task == 'fwd_fold':
            noisy_batch['t_trans'] = t_trans
            noisy_batch['t_rot'] = t_rot
            noisy_batch['t_seq'] = t_seq if self.task == 'cogen' else torch.ones_like(t_seq)

            trans_t, rotmats_t = self._corrupt_structure(noisy_batch, stage=stage)
            noisy_batch['trans_t'] = trans_t
            noisy_batch['rotmats_t'] = rotmats_t
            noisy_batch['aatypes_t'] = self._corrupt_aatypes(aatypes_1, t_seq, res_mask, diffuse_mask) if self.task == 'cogen' else aatypes_1
       
        elif self.task == 'inv_fold':
            noisy_batch['t_trans'] = torch.ones_like(t_trans)
            noisy_batch['t_rot'] = torch.ones_like(t_rot)
            noisy_batch['t_seq'] = t_seq

            noisy_batch['trans_t'] = trans_1
            noisy_batch['rotmats_t'] = rotmats_1
            noisy_batch['aatypes_t'] = self._corrupt_aatypes(aatypes_1, t_seq, res_mask, diffuse_mask)

        # self-conditioning information (stochastically over-written in FlowModule)
        noisy_batch['trans_sc'] = torch.zeros_like(trans_1)
        noisy_batch['aatypes_sc'] = torch.zeros_like(
            aatypes_1)[..., None].repeat(1, 1, self.num_tokens)
        return noisy_batch
    
    def sample_kappa(self, tau, component='rots', mode='train'):
        """ t = kappa(tau)
        Given tau [0,1], return the time step [0,1] to which to noise a given component.
        Enables differential noising of components during training and inference.

        IMPORTANT NOTE: 
        Call this during TRAINING to get time step to which to noise each component.
        Call this during INFERENCE to get the correct times to pass to the model for 
        the time-step embedding. Do NOT call the component-wise vector fields with 
        the output of this function, they are written in terms of tau, not t, and check 
        the desired noising schedule by themselves.
        """
        if component == 'rots':
            schedule = self._rots_cfg.train_schedule if mode == 'train' else self._rots_cfg.sample_schedule
            if schedule == 'exp':
                exp_rate = self._trans_cfg.train_exp_rate if mode == 'train' else self._trans_cfg.sample_exp_rate
            elif schedule == 'pow':
                power = self._rots_cfg.power
                
        elif component == 'trans':
            schedule = self._trans_cfg.train_schedule if mode == 'train' else self._trans_cfg.sample_schedule
            if schedule == 'exp':
                exp_rate = self._trans_cfg.train_exp_rate if mode == 'train' else self._trans_cfg.sample_exp_rate
            elif schedule == 'pow':
                power = self._trans_cfg.power

        elif component == 'seq':
            schedule = self._aatypes_cfg.train_schedule if mode == 'train' else self._aatypes_cfg.sample_schedule
            if schedule == 'exp':
                exp_rate = self._trans_cfg.train_exp_rate if mode == 'train' else self._trans_cfg.sample_exp_rate
            elif schedule == 'pow':
                power = self._aatypes_cfg.power

        else:
            raise ValueError(
                f'Unknown component {component}. Must be one of "rots", "trans", or "seq".')

        if schedule == 'exp':
            return 1 - torch.exp(-tau*exp_rate)
        elif schedule == 'pow':
            return tau**power
        elif schedule == 'linear':
            return tau
        else:
            raise ValueError(
                f'Invalid train schedule: {schedule}. Must be one of "exp" or "linear".')

    def _trans_vector_field(self, t, trans_1, trans_t):
        if self._trans_cfg.sample_schedule != 'linear':
            raise NotImplementedError(
                "Non-linear euler step for translations not implemented.")
        trans_vf = (trans_1 - trans_t) / (1 - t) # v_x in eq. (5) in FrameFlow paper
        return trans_vf

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        assert d_t >= 0
        trans_vf = self._trans_vector_field(t, trans_1, trans_t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        if self._rots_cfg.sample_schedule == 'linear':
            # for v_r in eq. (5) in FrameFlow paper
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == 'exp':
            # for v_r in eq. (7) in FrameFlow paper
            scaling = self._rots_cfg.sample_exp_rate
        else:
            raise ValueError(
                f'Unknown sample schedule {self._rots_cfg.sample_schedule}')
        return so3_utils.geodesic_t(scaling * d_t, rotmats_1, rotmats_t)

    def _regularize_step_probs(self, step_probs, aatypes_t):
        """ 
        Take model-derived probabilities over AAs in next step
        and set probability of remaining in current state to to 1-sum(rest).
        
        This programmatically enforces the various (j==x) cases in Multiflow 
        App. F, without requiring closed forms for each:
            * eq. (33) in F.1 and detailled-balanced form in F.1.1
            * (j=x) case under F.2 and detailled-balance form in F.2.1
        """
        batch_size, num_res, S = step_probs.shape
        device = step_probs.device
        assert aatypes_t.shape == (batch_size, num_res)
        
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)

        # untested scatter version of the below - cp also listing 6 (p. 44)
        # # indices of the current AA at each position (in flattened tensor)
        # indices = torch.arange(batch_size, device=device).repeat_interleave(num_res) * num_res + torch.arange(num_res, device=device).repeat(batch_size)
        # # replace those values with zero
        # step_probs.view(-1, S).scatter_(1, indices, 0.0)
        # # now replace them with 1-sum(rest)
        # step_probs.view(-1, S).scatter_(1, indices, (1.0 - torch.sum(step_probs, dim=-1)).flatten().unsqueeze(1))

        # TODO replace with torch._scatter as above (untested)
        step_probs[
            torch.arange(batch_size, device=device).repeat_interleave(num_res),
            torch.arange(num_res, device=device).repeat(batch_size),
            aatypes_t.long().flatten()
        ] = 0.0
        step_probs[
            torch.arange(batch_size, device=device).repeat_interleave(num_res),
            torch.arange(num_res, device=device).repeat(batch_size),
            aatypes_t.long().flatten()
        ] = 1.0 - torch.sum(step_probs, dim=-1).flatten()
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
        return step_probs

    def _aatypes_euler_step(self, d_t, t, logits_1, aatypes_t, is_final_step=False):
        """
        Given the current AA sequence, sample a new sequence using the model
        output logits, corresponding to an Euler step with our chosen
        interpolant (masking/uniform) and stochasticity (eta).

        See Multiflow paper, App. F.
        """
        batch_size, num_res, S = logits_1.shape
        assert aatypes_t.shape == (batch_size, num_res)
        assert t != 1, 'Transition probability undefined at t=1 due to division by zero!'
        device = logits_1.device

        eta = self._aatypes_cfg.noise
        
        if self._aatypes_cfg.interpolant_type == "masking":
            assert S == 21 # 20 AAs + <MASK>
            
            mask_one_hot = torch.zeros((S,), device=device)
            mask_one_hot[du.MASK_TOKEN_INDEX] = 1.0
            aatypes_t_is_mask = (aatypes_t == du.MASK_TOKEN_INDEX).view(batch_size, num_res, 1).float()

            # model-predicted probability of each AA at t=1 (pure data)
            # enforce zero probability of <MASK> token
            logits_1[:, :, du.MASK_TOKEN_INDEX] = -1e9 
            pt_x1_probs = F.softmax(logits_1 / self._aatypes_cfg.temp, dim=-1) # (B, D, S)

            # Multiflow, App. F.1.1 : Detailled-Balance Form of Transition Probability (p. 38, bottom)
            # (j!=x) case
                # NOTE: original first line below – incorrect, I think, for CURRENTLY UN-MASEKD positions
                # step_probs = d_t * pt_x1_probs * ((1+eta*t) / (1-t)) # (B, D, S)
            step_probs = d_t * pt_x1_probs * ((1+eta*t) / (1-t)) * aatypes_t_is_mask # (B, D, S)
            if not is_final_step: # don't allow re-masking in final step
                step_probs += d_t * (1 - aatypes_t_is_mask) * mask_one_hot.view(1, 1, -1) * eta
            
            # (j==x) case
            step_probs = self._regularize_step_probs(step_probs, aatypes_t)
            
        elif self._aatypes_cfg.interpolant_type == "uniform":
            assert S == 20 # 20 AAs
            assert aatypes_t.max() < 20, "No UNK tokens allowed in the uniform sampling step!"
            if is_final_step: # no noise in final step
                eta = 0.0

            # model-predicted probability of each AA at t=1 (pure data)
            pt_x1_probs = F.softmax(logits_1 / self._aatypes_cfg.temp, dim=-1) # (B, D, S)
            # model-predicted probability of the current AA at t=1 (pure data)
            pt_x1_eq_xt_prob = torch.gather(pt_x1_probs, dim=-1, index=aatypes_t.long().unsqueeze(-1)) # (B, D, 1)
            assert pt_x1_eq_xt_prob.shape == (batch_size, num_res, 1)

            # Multiflow, App. F.2.1 : Detailled-Balance Form of Transition Probability (p. 43, bottom)
            # (j!=x) case
            step_probs = d_t * (pt_x1_probs * ((1 + eta + eta*(S-1)*t) / (1-t)) + eta * pt_x1_eq_xt_prob)
            # (j==x) case
            step_probs = self._regularize_step_probs(step_probs, aatypes_t)
            
        else:
            raise ValueError(f"Unknown aatypes interpolant type {self._aatypes_cfg.interpolant_type}")

        return torch.multinomial(step_probs.view(-1, S), num_samples=1).view(batch_size, num_res)

    def _aatypes_euler_step_purity(self, d_t, t, logits_1, aatypes_t, is_final_step=False):
        """
        Given the current AA sequence, sample a new sequence using the model
        output logits and purity sampling to prioritise unmasking high-confidence 
        predictions. Corresponds to Euler step with MASKING INTERPOLANT and fixed 
        stochasticity eta under purity sampling.

        See Multiflow paper, App. F.1.2.
        """
        batch_size, num_res, S = logits_1.shape
        assert aatypes_t.shape == (batch_size, num_res)
        assert S == 21
        assert t != 1, 'Unmasking probability undefined at t=1 due to division by zero!'
        assert self._aatypes_cfg.interpolant_type == "masking"
        eta = self._aatypes_cfg.noise
        device = logits_1.device

        aatypes_t_is_mask = (aatypes_t == du.MASK_TOKEN_INDEX).float()

        # model-predicted probability of each AA at t=1 (pure data)
        # enforce zero probability of <MASK> token
        logits_1_wo_mask = logits_1[:, :, 0:-1] # (B, D, S-1)
        pt_x1_probs = F.softmax(logits_1_wo_mask / self._aatypes_cfg.temp, dim=-1) # (B, D, S-1)

        # sample new AA at every position, in case position will be unmasked
        unmasked_samples = torch.multinomial(pt_x1_probs.view(-1, S-1), num_samples=1).view(batch_size, num_res)

        # sort positions by confidence in prediction (purity), excluding already-unmasked positions
        max_logprob = torch.max(torch.log(pt_x1_probs), dim=-1)[0] # (B, D)
        max_logprob = max_logprob - (1-aatypes_t_is_mask) * 1e9
        sorted_max_logprobs_idcs = torch.argsort(max_logprob, dim=-1, descending=True) # (B, D)

        # choose how many positions to unmask for each instance in batch
        # App. F.1.2: #positions to unmask is binomially distributed w/ success prob dt*(1+eta*t)/(1−t) and #trials == #masekd_pos
        unmask_prob = (d_t * ( (1 + eta * t) / (1-t)).to(device)).clamp(max=1) # scalar
        num_masked = torch.count_nonzero(aatypes_t_is_mask, dim=-1).float()
        number_to_unmask = torch.binomial(count=num_masked, prob=unmask_prob)

        # pick the chosen number of positions to unmask, in descending order of confidence        
        # start by (B, D) list of top num_to_unmask indices (right-pad with first such index)
        D_grid = torch.arange(num_res, device=device).view(1, -1).repeat(batch_size, 1)
        mask1 = (D_grid < number_to_unmask.view(-1, 1)).float()
        inital_val_max_logprob_idcs = sorted_max_logprobs_idcs[:, 0].view(-1, 1).repeat(1, num_res)
        masked_sorted_max_logprobs_idcs = (mask1 * sorted_max_logprobs_idcs + (1-mask1) * inital_val_max_logprob_idcs).long()
        # set up (B, D) boolen grid of indices to unmask
        to_unmask = torch.zeros((batch_size, num_res), device=device)
        to_unmask.scatter_(dim=1, index=masked_sorted_max_logprobs_idcs, src=torch.ones((batch_size, num_res), device=device))
        # make sure instances where number_to_unmask == 0 don't unmask anything
        unmask_zero_row = (number_to_unmask == 0).view(-1, 1).repeat(1, num_res).float()
        to_unmask = to_unmask * (1 - unmask_zero_row)
        
        # pick positions to re-mask (of currently unmasked positions)
        u = torch.rand(batch_size, num_res, device=self._device)
        to_remask = (u < d_t * eta).float() * (1 - aatypes_t_is_mask)

        # sanity check
        assert torch.all((to_remask * to_unmask) == 0), "Overlap detected between to_remask and to_unmask"
        
        # unmask chosen positions
        aatypes_t = aatypes_t * (1 - to_unmask) + unmasked_samples * to_unmask
        
        # re-mask chosen positions
        if not is_final_step:
            aatypes_t = aatypes_t * (1 - to_remask) + du.MASK_TOKEN_INDEX * to_remask

        return aatypes_t

    def add_component_times(self, batch, tau):
        """ Update the batch dictionary with separate noising times for each component, 
        given integration time tau. Not called during training."""
        batch['t_trans'] = self.sample_kappa(tau, component='trans', mode='inference')
        batch['t_rot']   = self.sample_kappa(tau, component='rots', mode='inference')
        batch['t_seq']   = self.sample_kappa(tau, component='seq', mode='inference')
        if self.task == 'fwd_fold':
            # batch['t_seq'] = (1 - self._cfg.min_t) * torch.ones_like(batch['t_seq'])
            batch['t_seq'] = torch.ones_like(batch['t_seq'])
        if self.task == 'inv_fold': 
            batch['t_trans'] = (1 - self._cfg.min_t) * torch.ones_like(batch['t_trans'])
            batch['t_rot']   = (1 - self._cfg.min_t) * torch.ones_like(batch['t_rot'])
        return batch
    
    def update_self_conditioning(self, batch, initial_preds):
        """Update the batch dictionary with self-conditioning information from initial preds."""

        if self.gen_extent == 'partial':
            # both aa_types_1 and trans_1 are provided
            logits_1 = torch.nn.functional.one_hot(batch['aatypes_1'], 
                                                    num_classes=self.num_tokens).float()
            batch['aatypes_sc'] = _logits_diffuse_mask(initial_preds['pred_logits'], 
                                                        logits_1, 
                                                        batch['diffuse_mask'])
            batch['trans_sc'] = _trans_diffuse_mask(initial_preds['pred_trans'], 
                                                    batch['trans_1'], 
                                                    batch['diffuse_mask'])

        elif self.task == 'cogen' and self.gen_extent == 'full':
            # neither aatypes_1 nor trans_1 is provided
            batch['aatypes_sc'] = initial_preds['pred_logits']
            batch['trans_sc'] = initial_preds['pred_trans']

        elif self.task == 'fwd_fold' and self.gen_extent == 'full':
            # only aatypes_1 is provided
            logits_1 = torch.nn.functional.one_hot(batch['aatypes_1'], 
                                                    num_classes=self.num_tokens).float()
            batch['aatypes_sc'] = logits_1
            batch['trans_sc'] = initial_preds['pred_trans']
        
        elif self.task == 'inv_fold' and self.gen_extent == 'full': 
            # only trans_1 is provided
            batch['aatypes_sc'] = initial_preds['pred_logits']
            batch['trans_sc'] = batch['trans_1']

        return batch

    def _state_diffuse_mask(self, current_state, known_target_state, diffuse_mask):
        """ 
        During propagation, the current (noised) state is only to be used in
        the modalities and regions to be generated. Elsewhere, it should be 
        replaced by the provided target state.
        """
        trans_curr, rotmats_curr, aatypes_curr = current_state
        trans_1, rotmats_1, aatypes_1 = known_target_state

        if self.task == 'cogen' and self.gen_extent == 'partial':
            trans_curr   = _trans_diffuse_mask(trans_curr, trans_1, diffuse_mask)
            rotmats_curr = _rots_diffuse_mask(rotmats_curr, rotmats_1, diffuse_mask)
            aatypes_curr = _aatypes_diffuse_mask(aatypes_curr, aatypes_1, diffuse_mask)
        elif self.task == 'fwd_fold':
            aatypes_curr = aatypes_1
            if self.gen_extent == 'partial':
                trans_curr   = _trans_diffuse_mask(trans_curr, trans_1, diffuse_mask)
                rotmats_curr = _rots_diffuse_mask(rotmats_curr, rotmats_1, diffuse_mask)
        elif self.task == 'inv_fold':
            trans_curr = trans_1
            rotmats_curr = rotmats_1
            if self.gen_extent == 'partial':
                aatypes_curr = _aatypes_diffuse_mask(aatypes_curr, aatypes_1, diffuse_mask)
        
        return trans_curr, rotmats_curr, aatypes_curr

    # TODO: make this support parallel sampling
    def sample(self, batch, model):
        
        # batch must include:
            # res_idx       -> shape = num_batch, num_res
            # res_mask 
            # diffuse_mask
            # aatypes_1     -> for forward folding, or partial extent
            # rotmats_1     -> for inverse folding, or partial extent
            # trans_1       -> for inverse folding, or partial extent

        # preliminaries
        seq_itplt_type = self._aatypes_cfg.interpolant_type         # masking or uniform
        batch = {k: v.to(self._device) for k, v in batch.items()}
        num_batch, num_res = batch['res_mask'].shape

        # check inputs: are components required for chosen inference mode present?
        if self.gen_extent == 'full':
            assert torch.all(batch['diffuse_mask'] == batch['res_mask']), \
                "diffuse_mask and res_mask should not differ for full-antibody generation"
            if self.task == 'fwd_fold':
                assert batch['aatypes_1'] is not None
            elif self.task == 'inv_fold':
                assert batch['rotmats_1'] is not None
                assert batch['trans_1'] is not None
        elif self.gen_extent == 'partial':
            assert not (
                torch.all(batch['diffuse_mask'] == 1) or 
                torch.all(batch['diffuse_mask'] == 0)
            ), "diffuse_mask should not be all ones or all zeros for partial generation"
            assert batch['aatypes_1'] is not None
            assert batch['rotmats_1'] is not None
            assert batch['trans_1'] is not None

        # start with pure samples from the prior distributions (for each modality)
        # NOTE: here we use uniform on SO3 for rotations, training used IGSO3
        trans_0 = _centered_gaussian(num_batch, num_res, self._device) * du.NM_TO_ANG_SCALE
        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        if seq_itplt_type == "masking":
            aatypes_0 = _all_mask_token(num_batch, num_res, self._device)
        elif seq_itplt_type == "uniform":
            aatypes_0 = torch.randint_like(batch['res_mask'], low=0, high=self.num_tokens, device=self._device)
        else:
            raise ValueError(f"Unknown aatypes interpolant type {seq_itplt_type}")

        target_state = (batch['trans_1']   if 'trans_1'   in batch else None,
                        batch['rotmats_1'] if 'rotmats_1' in batch else None,
                        batch['aatypes_1'] if 'aatypes_1' in batch else None)
        current_state = (trans_0, rotmats_0, aatypes_0)
        # sets aatypes_0 to target state aatypes_1 for 'fwd_fold'
        current_state = self._state_diffuse_mask(current_state, target_state, batch['diffuse_mask'])

        # create self-conditioning information entries which need to be present in any case
        batch['trans_sc'] = torch.zeros_like(trans_0)
        batch['aatypes_sc'] = torch.zeros_like(aatypes_0)[..., None].repeat(1, 1, self.num_tokens)
        # if *actual* self-conditioning is enabled, update these entries
        if self._cfg.self_condition:
            # initial state
            batch['trans_t'], batch['rotmats_t'], batch['aatypes_t'] = current_state    
            
            # initial time(s)
            tau = torch.zeros((num_batch, 1), device=self._device)
            batch = self.add_component_times(batch, tau)                                     
            
            # run through model
            with torch.no_grad():
                model_out = model(batch)
            initial_preds = {'pred_logits': model_out['pred_logits'], 
                             'pred_trans': model_out['pred_trans']    }
            
            # use to create self-conditioning information
            batch = self.update_self_conditioning(batch, initial_preds)
        
        # set up time grid for integration
        # NOTE: closed-form Euler step for sequence is technically undefined at t=0 (see Multiflow App. F.1)
        ts = torch.linspace(0.0, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        # propagate system forward in time
        prot_traj, clean_traj = [current_state], []
        for t_2 in ts[1:]:
            
            # current time(s)
            tau = torch.ones((num_batch, 1), device=self._device) * t_1
            # t_seq == 1
            batch = self.add_component_times(batch, tau)
            
            # current state
            batch['trans_t'], batch['rotmats_t'], batch['aatypes_t'] = prot_traj[-1]
           
            # use model to predict a projected final state
            with torch.no_grad():
                model_out = model(batch)
            pred_trans_1    = model_out['pred_trans']    # projected positions at t=1
            pred_rotmats_1  = model_out['pred_rotmats']  # projected orientations at t=1
            pred_logits_1   = model_out['pred_logits']   # projected AA logits at t=1
            pred_aatypes_1  = model_out['pred_aatypes']  # argmax over projected AA logits at t=1
            pred_torsions_1 = model_out['pred_torsions']  # projected torsions at t=1

            if self.task == 'fwd_fold':
                pred_aatypes_1 = batch['aatypes_1']
            clean_traj.append((pred_trans_1, pred_rotmats_1, pred_aatypes_1, pred_torsions_1))

            ''' NOTE: Going from eq. (4.5) to eq. (6) in FrameFlow paper, we re-parametrised the loss
            s.t. the model predicts x_{t=1} and r_{t=1}, rather than the vector fields v_x and v_r.
            During sampling, when we actually *need* the vector fields, because
                        d/dt phi(x) = v(phi(x), t)
            we now back-compute them from the final-t prediction using eq. (5) [or (7)] in FrameFlow paper.

            The re-parametrisation was carried out to be able to easily apply the auxiliary structural 
            losses during training. See FlowModule->model_step() in models/flow_module.py for more 
            details. It also means that we can easily see how well we've learnt the OT path by looking 
            at the final-t projections from early sampling steps. We could potentially even truncate 
            the sampling after a few steps and use the final-t prediction, if it's good enough.

            Similarly, we make the model project final-state logits for the sequence part, training it with
            CE loss. During sampling, we now back-compute the transition pribabilities at the current time
            from these. See Multiflow eq. (33), and its equivalents in throughout App. F. '''
            
            # take single step towards projected final state
            d_t = t_2 - t_1
            trans_t_2   = self._trans_euler_step(d_t, t_1, pred_trans_1, batch['trans_t'])
            rotmats_t_2 = self._rots_euler_step(d_t, t_1, pred_rotmats_1,  batch['rotmats_t'])
            is_final_step = (t_2 == 1.0) 
            if self._aatypes_cfg.do_purity:
                aatypes_t_2 = self._aatypes_euler_step_purity(d_t, t_1, pred_logits_1, batch['aatypes_t'], 
                                                                is_final_step=is_final_step)
            else:
                aatypes_t_2 = self._aatypes_euler_step(d_t, t_1, pred_logits_1, batch['aatypes_t'], 
                                                        is_final_step=is_final_step)

            new_state = (trans_t_2, rotmats_t_2, aatypes_t_2)
            # sets aatypes_0 to target state aatypes_t_2 for 'fwd_fold'
            new_state = self._state_diffuse_mask(new_state, target_state, batch['diffuse_mask'])
            prot_traj.append(new_state)

            # update self-conditioning information for next step
            if self._cfg.self_condition:
                preds = {'pred_logits': pred_logits_1, 'pred_trans': pred_trans_1}
                batch = self.update_self_conditioning(batch, preds)
            
            # advance time and end iteration
            t_1 = t_2

        # We integrated to t=1, now make final prediction *at* t=1.
        
        # time(s)
        tau = torch.ones((num_batch, 1), device=self._device) * 1.0
        batch = self.add_component_times(batch, tau)
        
        # current state
        batch['trans_t'], batch['rotmats_t'], batch['aatypes_t'] = prot_traj[-1]

        # predict
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1    = model_out['pred_trans']    # projected positions at t=1
        pred_rotmats_1  = model_out['pred_rotmats']  # projected orientations at t=1
        pred_logits_1   = model_out['pred_logits']   # projected AA logits at t=1
        pred_aatypes_1  = model_out['pred_aatypes']  # argmax over projected AA logits at t=1
        pred_torsions_1 = model_out['pred_torsions']  # projected torsions at t=1
        
        # replace with known ground truth where available
        final_state = (pred_trans_1, pred_rotmats_1, pred_aatypes_1)
        # sets pred_aatypes_1 to target state aatypes_1 for 'fwd_fold'
        final_state = self._state_diffuse_mask(final_state, target_state, batch['diffuse_mask'])
        prot_traj.append(final_state)
        clean_traj.append(final_state + (pred_torsions_1,))

        # convert to coordinate representation and return
        atom37_traj, seq_traj = all_atom.convert_to_coords_and_detach(prot_traj, batch['res_mask'])
        clean_atom37_traj, clean_seq_traj = all_atom.convert_to_coords_and_detach(clean_traj, batch['res_mask'], torsions=True, batch=batch)
        
        return {'atom37_traj': atom37_traj,
                'seq_traj': seq_traj,
                'clean_atom37_traj': clean_atom37_traj,
                'clean_seq_traj': clean_seq_traj        }
