import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import argparse
import sys
sys.path.append('../')
from MLAb.metrics import prediction_metrics


def calculate_prediction_metrics(pdb_name, pred_dir, gt_file, model, do_sidechain='True'):
    
    if model == 'boltz': # sample a random prediction for boltz, simulate case of only one pred
        rand_idx = np.random.randint(0, 100)
        pred_file = os.path.join(pred_dir, pdb_name, f'{pdb_name}_model_{rand_idx:03d}.pdb')
    else:
        pred_file = os.path.join(pred_dir, pdb_name, f'{pdb_name}_model_000.pdb')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            metrics = prediction_metrics(pred_file, gt_file, do_sidechain=do_sidechain)
        except Exception as e:
            metrics = None
    return metrics

def main(pred_path, model_name, dset, do_sidechain='True', n_jobs=1):
    dataset = pd.read_csv('/vols/opig/users/spoendli/conformation_prediction/IgFM/metadata/ABB3_latest_metadata.csv', index_col=0) 
    dataset = dataset[dataset.latest_split == dset]

    pred_dir = f'{pred_path}/predictions'
    
    results = Parallel(n_jobs=n_jobs)(
            delayed(calculate_prediction_metrics)(
                row.pdb_name, pred_dir, row.raw_path, model_name, do_sidechain=do_sidechain
                ) for i, row in tqdm(dataset.iterrows(), total=dataset.shape[0], desc='Calculating metrics')
    )
    names = [dataset.pdb_name.values[i] for i in range(len(dataset.pdb_name.values)) if results[i] is not None]
    results = [result for result in results if result is not None] # remove None values
    results = pd.DataFrame(results, index=names)
    results.to_csv(f'{pred_path}/struc_pred_metrics_{dset}.csv')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Calculate structure prediction metrics')
    argparser.add_argument('--pred_path', type=str, required=True, help='Path to predictions')
    argparser.add_argument('--model_name', type=str, required=True, help='Model name')
    argparser.add_argument('--dset', type=str, required=True, help='Dataset name')  
    argparser.add_argument('--do_sidechain',  type=str, default='False', help='Enable sidechain processing')
    argparser.add_argument('--n_jobs', type=int, default=40, help='Number of parallel jobs')
    args = argparser.parse_args()
    main(args.pred_path, args.model_name, args.dset, do_sidechain=args.do_sidechain, n_jobs=args.n_jobs)
