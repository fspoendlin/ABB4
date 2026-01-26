from glob import glob
from abb4.data import utils as du
from abb4.analysis.ensemble_metrics import ensemble_metrics
import pickle
import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse


from glob import glob
from abb4.data import utils as du
from abb4.analysis.ensemble_metrics import ensemble_metrics
import pickle
import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse


def load_preds(name, pred_dir, filter_query=None):
    # filter prediction
    metadata = pd.read_csv(f'{pred_dir}/predictions_processed/metadata.csv', index_col=0)
    sele = metadata[metadata.pdb_name == name]
    if filter_query is not None:
        sele = sele.query(filter_query)
    pred_n_pass = sele.shape[0]

    file = '{}/predictions_processed/{}/{}.pkl'

    pred_atom37 = []
    pred_res_mask = []
    pred_imgt = []

    for model in sele.structure.values:
        with open(file.format(pred_dir, name, model), 'rb') as f:
            data = pickle.load(f)
        mask = data['bb_mask'] 
        pred_atom37.append(data['atom_positions']) # [mask.astype(bool)]) # mask applied when saving preds
        pred_res_mask.append((data['bb_mask'] == 1)[mask.astype(bool)])
        pred_imgt.append(data['residue_index'])

    pred_atom37 = torch.tensor(np.stack(pred_atom37))
    pred_res_mask = torch.tensor(np.stack(pred_res_mask))
    pred_imgt = torch.tensor(np.stack(pred_imgt))

    # equal probabilities for all states
    # pred_state_probas = torch.ones(pred_atom37.shape[0]) / pred_atom37.shape[0]

    return pred_atom37, pred_res_mask, pred_imgt, pred_n_pass

def format_simulation_id(simulation_id):
    if simulation_id.count('-') == 0:
        return simulation_id
    elif simulation_id.count('-') == 1:
        return simulation_id.split('_')[0].replace('-', '_')
    elif simulation_id.count('-') == 2:
        return f'{simulation_id.split("-")[0]}_{simulation_id.split("-")[1]}-{simulation_id.split("-")[2].split("_")[0]}'
    else:
        raise ValueError(f'Invalid simulation ID: {simulation_id}')

def load_gt(name, dataset_path, ds, probas_from_csv):
    # print(name)
    dataset = pd.read_csv(
        dataset_path,
        index_col=0, low_memory=False
    )

    dataset['processed_path'] = dataset['processed_path'].str.replace(
        '../multiflow_data/mAbs/',
        '/ceph/opig-shared/projects/spoendlin-EnsemblePrediction/'
    )
    dataset['simulation_ID'] = dataset['simulation_ID'].apply(format_simulation_id)
    if 'all_atom' in dataset_path:
        dataset = dataset[dataset.split == ds]
    else:
        dataset = dataset[dataset.latest_split == ds]
    gt_points = dataset[dataset['ABB3_set_map'] == name]
    if gt_points.shape[0] == 0: # no simulation for this datapoint
        return None, None, None, None
    if gt_points.simulation_ID.nunique() > 1: # if name links to two simulations
        select_gt_points = gt_points[gt_points.simulation_ID == name] # select the one with the same name
        if select_gt_points.shape[0] == 0: # select first simulation
            select_gt_points = gt_points[gt_points.simulation_ID == gt_points.simulation_ID.unique()[0]]
        gt_points = select_gt_points
    
    if probas_from_csv: # load probabilities from CSV
        gt_state_probas = torch.tensor(gt_points.joint_proba_norm.values)
    else: # no probabilities
        gt_state_probas = None

    gt_atom37 = []
    gt_res_mask = []
    gt_imgt = []
    gt_rmsd = []
    gt_rmsf = []

    for i, row in gt_points.iterrows():
        processed_feats = du.load_rigids(row.processed_path, return_feats=True)[3]

        gt_atom37.append(processed_feats['atom_positions'])
        gt_res_mask.append(processed_feats['bb_mask'])
        gt_imgt.append(processed_feats['residue_index'])
        gt_rmsd.append(processed_feats['rmsd'].astype(float))
        gt_rmsf.append(processed_feats['rmsf'].astype(float))

    gt_atom37 = torch.tensor(np.stack(gt_atom37))
    gt_res_mask = torch.tensor(np.stack(gt_res_mask))
    gt_imgt = torch.tensor(np.stack(gt_imgt))
    gt_rmsd = torch.tensor(gt_rmsd[0]) # shape [9, 3]
    gt_rmsf = torch.tensor(gt_rmsf[0].T) # shape [N, 3]
    
    return gt_atom37, gt_res_mask, gt_imgt, gt_state_probas, gt_rmsd, gt_rmsf

def mask_extra_residues(
        gt_atom37, gt_res_mask, gt_imgt, gt_rmsf,
        pred_atom37, pred_res_mask, pred_imgt
    ):
    # find common imgt positions
    common_imgt = pred_imgt[0][torch.isin(pred_imgt[0], gt_imgt[0])]
    pred_mask = torch.isin(pred_imgt[0], common_imgt)
    gt_mask = torch.isin(gt_imgt[0], common_imgt)

    # apply mask residues
    gt_atom37 = gt_atom37[:, gt_mask, :, :]
    gt_res_mask = gt_res_mask[:, gt_mask]
    gt_imgt = gt_imgt[:, gt_mask]
    gt_rmsf = gt_rmsf[gt_mask] # shape [N, 3]

    pred_atom37 = pred_atom37[:, pred_mask, :, :]
    pred_res_mask = pred_res_mask[:, pred_mask]
    pred_imgt = pred_imgt[:, pred_mask]

    return (gt_atom37, gt_res_mask, gt_imgt, gt_rmsf,
            pred_atom37, pred_res_mask, pred_imgt)

def get_metrics_from_name(name, pred_dir, dataset_path, ds, probas_from_csv=False, filter_query=None):
    try:
        pred_atom37, pred_res_mask, pred_imgt, pred_n_pass = load_preds(name, pred_dir, filter_query)
        gt_atom37, gt_res_mask, gt_imgt, gt_state_probas, gt_rmsd, gt_rmsf = load_gt(name, dataset_path, ds, probas_from_csv)
    except Exception as e:
        print(f'Error loading data for {name}: {e}')
        raise ValueError('error loading data')

    if gt_atom37 is None:
        print(f'No ground truth data for {name}')
        return None
    else:
        # remove imgt residues missing in either gt or pred
        masked = mask_extra_residues(
            gt_atom37, gt_res_mask, gt_imgt, gt_rmsf,
            pred_atom37, pred_res_mask, pred_imgt
        )
        gt_atom37, gt_res_mask, gt_imgt, gt_rmsf, pred_atom37, pred_res_mask, pred_imgt = masked    
        
        if pred_atom37.shape[1] != gt_atom37.shape[1]:
            print('atom_missmatch')
            return None
        try:
            metrics = ensemble_metrics(
                gt_atom37, pred_atom37, gt_res_mask, gt_imgt, gt_state_probas, gt_rmsd, gt_rmsf,
            )
            metrics['pred_n_pass'] = pred_n_pass
        except Exception as e:
            print(f'Error calculating metrics for {name}: {e}')
            return None
        return metrics

def main(model, dataset_path, dset, probas_from_csv=False, filter_query=None, n_cores=10, all_atom_sims=False):
    pred_dir = f'/ceph/opig-shared/users/spoendli/conformation_prediction/conformation_prediction/analysis/conformation_prediction_benchmark/{model}'
    if os.path.exists(pred_dir) is False:
        pred_dir = f'/ceph/opig-shared/users/spoendli/conformation_prediction/conformation_prediction/analysis/IgFM_predictions/{model}'
    if os.path.exists(pred_dir) is False:
        pred_dir = f'/ceph/opig-shared/projects/spoendlin-EnsemblePrediction/EnsembleStructures/IgFM_predictions/{model}'
    print(f'Loading predictions from {pred_dir}')

    dataset = pd.read_csv(
        dataset_path,
        index_col=0, low_memory=False
    )
    dataset.drop_duplicates(subset=['ABB3_set_map'], inplace=True)
    if all_atom_sims:
        dataset = dataset[dataset.split == dset]
    else:
        dataset = dataset[dataset.latest_split == dset]
    names = dataset.ABB3_set_map.values

    metrics = Parallel(n_jobs=n_cores)(
        delayed(get_metrics_from_name)(
            name, pred_dir, dataset_path, dset, probas_from_csv=probas_from_csv, filter_query=filter_query
            ) for name in tqdm(names)
    )

    # remove None values
    names = [names[i] for i in range(len(names)) if metrics[i] is not None]
    metrics = [metric for metric in metrics if metric is not None]

    df = pd.DataFrame(metrics, index=names)
    suffix = '_random' if 'random' in dataset_path else ''
    prefix = '_aa' if all_atom_sims else ''
    df.to_pickle(f'{pred_dir}/{dset}{prefix}_ensemble_metrics{suffix}.pkl')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, required=True)
    argparser.add_argument('--dset', type=str, required=True)
    # argparser.add_argument('--filter_query', type=str, default='dc <= 24.3 & dc >= 20.4')
    argparser.add_argument('--n_cores', type=int, default=10)
    argparser.add_argument('--dataset_path', type=str, default='/ceph/opig-shared/users/spoendli/conformation_prediction/IgFM/metadata/ABB3_latest_CDRsim_metadata_random_ceph_paths.csv')
    argparser.add_argument('--probas_from_csv', action='store_true', default=False)
    argparser.add_argument('--all_atom_sims', action='store_true', default=False)
    args = argparser.parse_args()
    args.filter_query = None

    main(
        model=args.model,
        dataset_path=args.dataset_path,
        dset=args.dset,
        probas_from_csv=args.probas_from_csv,
        filter_query=args.filter_query,
        n_cores=args.n_cores,
        all_atom_sims=args.all_atom_sims
    )
