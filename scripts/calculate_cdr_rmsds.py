'''Script to calculate CDR RMSDs for predicted antibody structure ensembles'''
from SPACE2.exhaustive_clustering import get_distance_matrices
from SPACE2.util import reg_def
import numpy as np
import pandas as pd
from glob import glob
import os
from tqdm import tqdm
import argparse

arg_parser = argparse.ArgumentParser(description="Calculate CDR RMSDs for predicted antibody structure ensembles")
arg_parser.add_argument('--pred_path', type=str, required=True, help='Path to the predicted PDB files directory')
arg_parser.add_argument('--n_jobs', type=int, default=20, help='Number of parallel jobs to run')
args = arg_parser.parse_args()

length_tolerance = np.array([1])
def cdr_rmsd(cdr, files, n_jobs=10):
    dmat = get_distance_matrices(files, selection=[reg_def[cdr]], anchors=[reg_def[f"fw{cdr[-2].upper()}"]], d_metric='rmsd',
                      length_tolerance=length_tolerance, n_jobs=n_jobs)
    
    mat = dmat[list(dmat.keys())[0]][1]
    mask = np.triu(np.ones_like(mat), k=1).astype(bool)
    masked_mat = np.where(mask, mat, np.nan)

    return np.array([np.nanmean(masked_mat), np.nanmax(masked_mat), np.nanmin(masked_mat), np.nanstd(masked_mat)])


def rmsd4path(pdb_dir, n_jobs=20):
    files = glob(f'{pdb_dir}/*.pdb')
    results = {'mean': {}, 'max': {}, 'min': {}, 'std': {}}
    for cdr in tqdm(['CDRH1', 'CDRH2', 'CDRH3', 'CDRL1', 'CDRL2', 'CDRL3'], desc='Calculating CDR RMSDs', total=6):
        mean, max, min, std = cdr_rmsd(cdr, files, n_jobs=n_jobs)
        results['mean'][cdr] = mean
        results['max'][cdr] = max
        results['min'][cdr] = min
        results['std'][cdr] = std

    return results

def main(pred_path, n_jobs=20):
    dirs = [d for d in glob(f"{pred_path}/*") if os.path.isdir(d)]

    results = {}
    for dir in dirs:
        print(f"Processing directory: {dir}")
        results[os.path.basename(dir)] = rmsd4path(dir, n_jobs=n_jobs)

        df = pd.DataFrame.from_dict({(i,j): results[i][j] 
                            for i in results.keys()
                            for j in results[i].keys()}, orient='index')

        df.to_csv(os.path.join(pred_path, 'cdr_rmsds.csv'))

if __name__ == "__main__":
    main(args.pred_path, n_jobs=20)
