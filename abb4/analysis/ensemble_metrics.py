'''Calculate ensemble validation metrics'''
import numpy as np
import torch
import ot as pot
from openfold.utils.superimposition import superimpose


chain_term = 1000
reg_def = {'cdr1': np.arange(27, 38+1), 'cdr2': np.arange(56, 65+1), 'cdr3': np.arange(105, 117+1)}
reg_def['cdrs'] = np.concatenate([reg_def['cdr1'], reg_def['cdr2'], reg_def['cdr3']])
reg_def['fw'] = np.array([x for x in range(1, 128+1) if x not in reg_def['cdrs']])


def rmsd_mat2accuracy_metrics(rmsd, gt_state_probas, topn=10):
    # recall: distance from each GT to closest pred
    recall_full = torch.min(rmsd, dim=0).values
    recall = recall_full.mean()
    top_n_recall = torch.topk(rmsd, k=topn, largest=False, dim=0).values.mean(dim=0)

    # precision: distance from each pred to closest GT
    precision_full = torch.min(rmsd, dim=1).values
    precision = precision_full.mean()

    # distributional accuracy
    rmsd = rmsd.numpy()
    uniform_dist = np.ones(rmsd.shape[0]) / rmsd.shape[0]
    if not gt_state_probas: # use uniform if nonw
        gt_state_probas = np.ones(rmsd.shape[1]) / rmsd.shape[1]
    else:
        gt_state_probas = gt_state_probas.numpy()

    W = pot.emd2(
        uniform_dist,
        gt_state_probas,
        rmsd
    )
    return recall, recall_full, top_n_recall, precision, precision_full, W

def domain_accuracy_metrics(exp_gt_bb_pos, exp_pred_bb_pos, exp_res_mask, num_gt_batch, num_pred_batch, gt_state_probas, topn=10):
    _, rmsd = superimpose(exp_gt_bb_pos, exp_pred_bb_pos, mask=exp_res_mask)
    rmsd = rmsd.reshape(num_pred_batch, num_gt_batch) # [pred idx, gt idx]

    return rmsd_mat2accuracy_metrics(rmsd, gt_state_probas, topn=topn)

def rmsd_mat2rmsd_metrics(pred_rmsd, gt_rmsd, gt_state_probas):
    # max distance between any two preds
    pred_rmsd_max = torch.max(pred_rmsd)
    pred_rmsd = pred_rmsd.fill_diagonal_(float('nan'))
    pred_rmsd_mean = torch.nanmean(pred_rmsd)

    # max distance between any two gts
    gt_rmsd_max = torch.max(gt_rmsd)
    if gt_state_probas: # weighted mean
        gt_joined_proba = gt_state_probas[None, :] * gt_state_probas[:, None] # already normalised
        gt_rmsd_mean = torch.sum(gt_rmsd * gt_joined_proba)
    else: # regular mean
        gt_rmsd = gt_rmsd.fill_diagonal_(float('nan'))
        gt_rmsd_mean = torch.nanmean(gt_rmsd)

    return (pred_rmsd_max,
            pred_rmsd_mean,
            gt_rmsd_max,
            gt_rmsd_mean,
    )

def domain_rmsd_metrics(
        exp_gt_bb_pos1, exp_gt_bb_pos2,
        exp_pred_bb_pos1, exp_pred_bb_pos2,
        mask, num_gt_batch, num_pred_batch,
        gt_state_probas):
    # pred rmsd matrix
    _, pred_rmsd = superimpose(exp_pred_bb_pos1, exp_pred_bb_pos2, mask=mask.repeat(exp_pred_bb_pos1.shape[0], 1))
    pred_rmsd = pred_rmsd.reshape(num_pred_batch, num_pred_batch)

    # gt rmsd matrix
    _, gt_rmsd_frames = superimpose(exp_gt_bb_pos1, exp_gt_bb_pos2, mask=mask.repeat(exp_gt_bb_pos1.shape[0], 1))
    gt_rmsd_frames = gt_rmsd_frames.reshape(num_gt_batch, num_gt_batch)

    return rmsd_mat2rmsd_metrics(pred_rmsd, gt_rmsd_frames, gt_state_probas)

def rmsd_mat(region1, region2, num_batch1, num_batch2):
    sq_dist = (torch.sum(torch.pow((region1 - region2),2), dim=2))
    rmsds = torch.sqrt(torch.nanmean(sq_dist, dim=1))
    return rmsds.reshape(num_batch1, num_batch2)

def calculate_CA_rmsf(pred_atom37, gt_atom37, gt_state_probas, gt_rmsf_sim, region_mask, superimpose_mask):
    # convert to CA coordinate mask
    # only select every 3rd along axis -1
    region_mask = region_mask.bool()
    CA_region_mask = region_mask.reshape(1, -1, 3)[0, :, 0]
    superimpose_mask = superimpose_mask.reshape(1, -1, 3)[:, :, 0]

    # pred
    pred_CA_coords = pred_atom37[:, :, 1, :]
    # superimpose on first frame
    pred_CA_coords,  _ = superimpose(
        pred_CA_coords[0][None, ...].repeat(pred_CA_coords.shape[0],1,1),
        pred_CA_coords,
        superimpose_mask.repeat(pred_atom37.shape[0], 1)
    )
    assert (pred_CA_coords.shape[1] == CA_region_mask.shape[0] and
            pred_CA_coords.shape[1] == superimpose_mask.shape[1])
    pred_CA_mean = torch.mean(pred_CA_coords, axis=0)
    pred_sq_dist = torch.sum(torch.pow(pred_CA_coords - pred_CA_mean[None, :, :], 2), axis=-1)
    pred_rmsf = torch.sqrt(torch.mean(pred_sq_dist, axis=0))
    pred_rmsf = pred_rmsf[CA_region_mask]

    # gt
    gt_CA_coords = gt_atom37[:, :, 1, :]
    # superimpose on first frame
    gt_CA_coords,  _ = superimpose(
        gt_CA_coords[0][None, ...].repeat(gt_CA_coords.shape[0],1,1),
        gt_CA_coords,
        superimpose_mask.repeat(gt_atom37.shape[0], 1)
    )
    assert (gt_CA_coords.shape[1] == CA_region_mask.shape[0] and
            gt_CA_coords.shape[1] == superimpose_mask.shape[1])

    if gt_state_probas: # weighted RMSF
        gt_CA_mean = torch.sum(gt_state_probas[:, None, None] * gt_CA_coords, axis=0)
        gt_sq_dist = torch.sum(torch.pow(gt_CA_coords - gt_CA_mean[None, :, :], 2), axis=-1)
        gt_rmsf_frames = torch.sqrt(torch.sum(gt_sq_dist * gt_state_probas[:, None], axis=0))
    else: # regular RMSF
        gt_CA_mean = torch.mean(gt_CA_coords, axis=0)
        gt_sq_dist = torch.sum(torch.pow(gt_CA_coords - gt_CA_mean[None, :, :], 2), axis=-1)
        gt_rmsf_frames = torch.sqrt(torch.mean(gt_sq_dist, axis=0))
    gt_rmsf_frames = gt_rmsf_frames[CA_region_mask]

    gt_rmsf_sim = gt_rmsf_sim[CA_region_mask]

    # rmse
    rmsf_rmse_frames = torch.sqrt(torch.mean(torch.pow(pred_rmsf - gt_rmsf_frames, 2)))
    rmsf_rmse_sim = torch.sqrt(torch.mean(torch.pow(pred_rmsf - gt_rmsf_sim, 2)))

    return (pred_rmsf.max(), pred_rmsf.mean(),
            gt_rmsf_frames.max(), gt_rmsf_frames.mean(), rmsf_rmse_frames,
            gt_rmsf_sim.max(), gt_rmsf_sim.mean(), rmsf_rmse_sim)

def write_ensemble_metrics(recall, recall_full, top_n_recall, precision, precision_full, W,
                           max_pred_rmsd, mean_pred_rmsd,
                           max_gt_rmsd_frames, mean_gt_rmsd_frames,
                           max_gt_rmsd_sim, mean_gt_rmsd_sim,
                           max_pred_rmsf, mean_pred_rmsf,
                           max_gt_rmsf_frames, mean_gt_rmsf_frames, rmsf_rmse_frames,
                           max_gt_rmsf_sim, mean_gt_rmsf_sim, rmsf_rmse_sim,
                           region, metrics
    ):
    metrics[f'{region}_recall'] = recall.item()
    metrics[f'{region}_recall_full'] = recall_full.tolist()
    metrics[f'{region}_top_n_recall'] = top_n_recall.tolist()
    metrics[f'{region}_precision'] = precision.item()
    metrics[f'{region}_precision_full'] = precision_full.tolist()
    metrics[f'{region}_W'] = W
    metrics[f'{region}_max_pred_rmsd'] = max_pred_rmsd.item()
    metrics[f'{region}_mean_pred_rmsd'] = mean_pred_rmsd.item()
    metrics[f'{region}_max_gt_rmsd'] = max_gt_rmsd_frames.item()
    metrics[f'{region}_mean_gt_rmsd'] = mean_gt_rmsd_frames.item()
    metrics[f'{region}_max_gt_rmsd_sim'] = max_gt_rmsd_sim.item()
    metrics[f'{region}_mean_gt_rmsd_sim'] = mean_gt_rmsd_sim.item()
    metrics[f'{region}_max_pred_rmsf'] = max_pred_rmsf.item()
    metrics[f'{region}_mean_pred_rmsf'] = mean_pred_rmsf.item()
    metrics[f'{region}_max_gt_rmsf'] = max_gt_rmsf_frames.item()
    metrics[f'{region}_mean_gt_rmsf'] = mean_gt_rmsf_frames.item()
    metrics[f'{region}_rmsf_rmse'] = rmsf_rmse_frames.item()
    metrics[f'{region}_max_gt_rmsf_sim'] = max_gt_rmsf_sim.item()
    metrics[f'{region}_mean_gt_rmsf_sim'] = mean_gt_rmsf_sim.item()
    metrics[f'{region}_rmsf_rmse_sim'] = rmsf_rmse_sim.item()
    return metrics


region2rmsdIdx = {
    'global': 0,
    'h_chain': 1,
    'l_chain': 2,
    'h': {
        'cdr1': 3,
        'cdr2': 4,
        'cdr3': 5,
    },
    'l': {
        'cdr1': 6,
        'cdr2': 7,
        'cdr3': 8,
    }
}


def ensemble_metrics(gt_atom37, pred_atom37, res_mask, imgt, gt_state_probas, gt_rmsd_sim, gt_rmsf_sim, topn=10):
    num_gt_batch = gt_atom37.shape[0]
    num_pred_batch = pred_atom37.shape[0]
    metrics = {}

    # reshape inputs
    expand_imgt = imgt[0].unsqueeze(0).unsqueeze(2).repeat(1,1,3).reshape(1,-1)
    expand_res_mask = res_mask[0].unsqueeze(0).unsqueeze(2).repeat(1,1,3).reshape(1,-1)
    H_chain_mask = (expand_imgt < 1000).int() * expand_res_mask
    L_chain_mask = (expand_imgt >= 1000).int() * expand_res_mask

    # copy tensors to correct dimensions for rmsd matrix calculation
    gt_bb_pos = gt_atom37[:,:,:3, :].reshape(num_gt_batch,-1,3)
    pred_bb_pos = pred_atom37[:,:,:3, :].reshape(num_pred_batch,-1,3)

    pred_idx, gt_idx = torch.where(
        torch.ones(num_pred_batch, num_gt_batch))
    expand_pred_bb_pos = pred_bb_pos[pred_idx]
    expand_gt_bb_pos = gt_bb_pos[gt_idx]

    pred2a_idx, pred2b_idx = torch.where(
        torch.ones(num_pred_batch, num_pred_batch))
    expand_pred_bb_pos2a = pred_bb_pos[pred2a_idx]
    expand_pred_bb_pos2b = pred_bb_pos[pred2b_idx]

    gt2a_idx, gt2b_idx = torch.where(
        torch.ones(num_gt_batch, num_gt_batch))
    expand_gt_bb_pos2a = gt_bb_pos[gt2a_idx]
    expand_gt_bb_pos2b = gt_bb_pos[gt2b_idx]

    # chain metrics
    for region, mask in zip(['global', 'h_chain', 'l_chain'], [expand_res_mask, H_chain_mask, L_chain_mask]):

        recall, recall_full, top_n_recall, precision, precision_full, W = domain_accuracy_metrics(
            expand_gt_bb_pos, expand_pred_bb_pos, mask.repeat(expand_gt_bb_pos.shape[0], 1), num_gt_batch, num_pred_batch,
            gt_state_probas, topn=topn
        )

        max_pred_rmsd, mean_pred_rmsd, max_gt_rmsd_frames, mean_gt_rmsd_frames = domain_rmsd_metrics(
            expand_gt_bb_pos2a, expand_gt_bb_pos2b,
            expand_pred_bb_pos2a, expand_pred_bb_pos2b,
            mask, num_gt_batch, num_pred_batch,
            gt_state_probas,
        )
        max_gt_rmsd_sim = gt_rmsd_sim[region2rmsdIdx[region], 0] # index 0 for max
        mean_gt_rmsd_sim = gt_rmsd_sim[region2rmsdIdx[region], 1] # index 1 for mean

        # rmsf
        idx = 0 if region == 'global' else 2 # 0 gobal aligned, 2 chain aligned
        gt_rmsf_sel = gt_rmsf_sim[:, idx] # shape [N, 3]
        max_pred_rmsf, mean_pred_rmsf, max_gt_rmsf_frames, mean_gt_rmsf_frames, rmsf_rmse_frames, max_gt_rmsf_sim, mean_gt_rmsf_sim, rmsf_rmse_sim = calculate_CA_rmsf(
            pred_atom37, gt_atom37, gt_state_probas, gt_rmsf_sel, mask, mask
        )

        metrics = write_ensemble_metrics(
            recall,
            recall_full,
            top_n_recall,
            precision,
            precision_full,
            W,
            max_pred_rmsd,
            mean_pred_rmsd,
            max_gt_rmsd_frames,
            mean_gt_rmsd_frames,
            max_gt_rmsd_sim,
            mean_gt_rmsd_sim,
            max_pred_rmsf,
            mean_pred_rmsf,
            max_gt_rmsf_frames,
            mean_gt_rmsf_frames,
            rmsf_rmse_frames,
            max_gt_rmsf_sim,
            mean_gt_rmsf_sim,
            rmsf_rmse_sim,
            region,
            metrics
        )

    # cdr metrics
    for chain, mask in zip(['h', 'l'], [H_chain_mask, L_chain_mask]):
        # align on whole chain
        pred_imposed_on_gt, _ = superimpose(expand_gt_bb_pos, expand_pred_bb_pos, mask=mask.repeat(expand_gt_bb_pos.shape[0], 1))
        pred_imposed_on_pred, _ = superimpose(expand_pred_bb_pos2a, expand_pred_bb_pos2b, mask=mask.repeat(expand_pred_bb_pos2a.shape[0], 1))
        gt_imposed_on_gt, _ = superimpose(expand_gt_bb_pos2a, expand_gt_bb_pos2b, mask=mask.repeat(expand_gt_bb_pos2a.shape[0], 1))

        for region, resi in reg_def.items():
            if region == 'fw' or region == 'cdrs': # ignore as no GT RMSD for these regions
                continue
            resi = torch.tensor(resi)
            if chain == 'l':
                resi = resi + chain_term
            elif chain == 'h':
                resi = resi
            region_mask = torch.isin(expand_imgt, resi)

            # gt to pred
            gt = expand_gt_bb_pos.clone()
            gt[~region_mask.repeat(gt.shape[0], 1)] = torch.nan
            pred_on_gt = pred_imposed_on_gt.clone()
            pred_on_gt[~region_mask.repeat(pred_on_gt.shape[0], 1)] = torch.nan

            rmsds = rmsd_mat(pred_on_gt, gt, num_pred_batch, num_gt_batch)

            # pred to pred
            pred = expand_pred_bb_pos2a.clone()
            pred[~region_mask.repeat(pred.shape[0], 1)] = torch.nan
            pred_on_pred = pred_imposed_on_pred.clone()
            pred_on_pred[~region_mask.repeat(pred_on_pred.shape[0], 1)] = torch.nan

            pred_rmsd = rmsd_mat(pred, pred_on_pred, num_pred_batch, num_pred_batch)

            # gt to gt
            gt = expand_gt_bb_pos2a.clone()
            gt[~region_mask.repeat(gt.shape[0], 1)] = torch.nan
            gt_on_gt = gt_imposed_on_gt.clone()
            gt_on_gt[~region_mask.repeat(gt_on_gt.shape[0], 1)] = torch.nan

            gt_rmsd = rmsd_mat(gt, gt_on_gt, num_gt_batch, num_gt_batch)

            recall, recall_full, top_n_recall, precision, precision_full, W  = rmsd_mat2accuracy_metrics(rmsds, gt_state_probas, topn=topn)
            max_pred_rmsd, mean_pred_rmsd, max_gt_rmsd_frames, mean_gt_rmsd_frames = rmsd_mat2rmsd_metrics(
                pred_rmsd, gt_rmsd, gt_state_probas,
            )
            max_gt_rmsd_sim = gt_rmsd_sim[region2rmsdIdx[chain][region], 0] # index 0 for max
            mean_gt_rmsd_sim = gt_rmsd_sim[region2rmsdIdx[chain][region], 1] # index 1 for mean

            # rmsf
            gt_rmsf_sel = gt_rmsf_sim[:, 2] # 2 chain aligned
            max_pred_rmsf, mean_pred_rmsf, max_gt_rmsf_frames, mean_gt_rmsf_frames, rmsf_rmse_frames, max_gt_rmsf_sim, mean_gt_rmsf_sim, rmsf_rmse_sim = calculate_CA_rmsf(
               pred_atom37, gt_atom37, gt_state_probas, gt_rmsf_sel, region_mask, mask
            )

            metrics = write_ensemble_metrics(
                recall,
                recall_full,
                top_n_recall,
                precision,
                precision_full,
                W,
                max_pred_rmsd,
                mean_pred_rmsd,
                max_gt_rmsd_frames,
                mean_gt_rmsd_frames,
                max_gt_rmsd_sim,
                mean_gt_rmsd_sim,
                max_pred_rmsf,
                mean_pred_rmsf,
                max_gt_rmsf_frames,
                mean_gt_rmsf_frames,
                rmsf_rmse_frames,
                max_gt_rmsf_sim,
                mean_gt_rmsf_sim,
                rmsf_rmse_sim,
                f'{chain}_{region}',
                metrics
            )

    return metrics
