'''Script to calculate CDR RMSD statistics for predicted ensembles.'''
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from SPACE2.exhaustive_clustering import get_distance_matrices
from helper_functions.reg_def_mAbs import *
from glob import glob
from pathlib import Path
import os
import argparse
from joblib import parallel_backend

CDRs = ['CDRH1', 'CDRH2', 'CDRH3', 'CDRL1', 'CDRL2', 'CDRL3']

def process_single_row(pdb_name, pred_dir, CDR):
    """Process a single row for flexibility calculation"""
    files = glob(f'{pred_dir}{pdb_name}/*.pdb')
    # assert len(files) == 100, f'Expected 100 files, found {len(files)} for {pdb_name}'
    if len(files) != 100:
        print(f'Warning: Expected 100 files, found {len(files)} for {pdb_name}.')

    try:
        mats = get_distance_matrices(files, anchors=[reg_def[f'fw{CDR[3]}']], selection=[reg_def[CDR]], length_tolerance=np.ones(1), n_jobs=1)
        mat = mats[list(mats.keys())[0]][1]
        upper_triu = mat[np.triu_indices_from(mat, k=1)]
    except Exception as e:
        print(f'Error processing {pdb_name} for {CDR}: {e}')
        return pdb_name, {
            'mean': np.nan,
            'std': np.nan,
            'median': np.nan,
            'max': np.nan,
            'per95': np.nan,
        }

    return pdb_name, {
        'mean': np.mean(upper_triu),
        'std': np.std(upper_triu),
        'median': np.median(upper_triu),
        'max': np.max(upper_triu),
        'per95': np.percentile(upper_triu, 95),
    }

def get_flexibility_stats(pred_dir, n_jobs=30):
    pdb_names = [os.path.basename(x) for x in glob(f'{pred_dir}/*')]

    for i, CDR in enumerate(CDRs):
        # Parallelize the processing
        with parallel_backend("loky"):
            results_list = Parallel(n_jobs=n_jobs)(
                delayed(process_single_row)(pdb_name, pred_dir, CDR) 
                for pdb_name in tqdm(pdb_names, total=len(pdb_names), desc=f'Processing {CDR}, iteration {i+1}/{len(CDRs)}')
            )

        # results_list = [process_single_row(pdb_name, pred_dir, CDR) for pdb_name in tqdm(pdb_names, total=len(pdb_names), desc=f'Processing {CDR}, iteration {i+1}/{len(CDRs)}')]
        
        # Convert results to dictionary
        results = {key: value for key, value in results_list}

        results = pd.DataFrame.from_dict(results, orient='index')
        results.to_csv(f'{str(Path(pred_dir).parent)}/{CDR}_flexibility.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate CDR flexibility statistics from predicted ensembles.')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory containing predicted ensembles.')
    parser.add_argument('--n_jobs', type=int, default=30, help='Number of parallel jobs to run.')
    
    args = parser.parse_args()
    
    print(f'Calculating flexibility statistics for predictions in {args.pred_dir}')
    get_flexibility_stats(args.pred_dir, n_jobs=args.n_jobs)
