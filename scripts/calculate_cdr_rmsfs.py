import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
from glob import glob
from MDAnalysis.analysis import align
import argparse
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', type=str, required=True, help='Path to directories with PDB files')
parser.add_argument('--n_jobs', type=int, default=1, help='Number of processes for directory-level parallelism')
parser.add_argument('--output_csv', type=str, default='cdr_rmsfs.csv', help='Output CSV file name for the results')
args = parser.parse_args()

cdr2range = {
    'CDR1': (27, 38),
    'CDR2': (56, 65),
    'CDR3': (105, 117),
}

def calculate_CDR_rmsf(pdb_files, chain, cdr):

    u = mda.Universe(pdb_files[0], pdb_files)
    reference = mda.Universe(pdb_files[0])

    aligner = align.AlignTraj(
        u,
        reference,
        select=f"chainID {chain} and name CA",
        in_memory=True
    )
    aligner.run()

    sel = u.select_atoms(f"chainID {chain} and name CA and resid {cdr2range[cdr][0]}-{cdr2range[cdr][1]}")  # CDR residues
    rmsf = rms.RMSF(sel).run()
    rmsf_values = rmsf.results.rmsf

    return rmsf_values


def process_directory(d):
    code = os.path.basename(d)
    pdb_files = glob(f'{d}/*.pdb')

    if len(pdb_files) == 0:
        return None

    result = {'code': code}
    for cdr in ['CDRH1', 'CDRH2', 'CDRH3', 'CDRL1', 'CDRL2', 'CDRL3']:
        chain = cdr[3]
        rmsf_values = calculate_CDR_rmsf(pdb_files, chain, cdr[:-2] + cdr[-1])
        result[f'{cdr}_mean'] = np.mean(rmsf_values)
        result[f'{cdr}_std'] = np.std(rmsf_values)
        result[f'{cdr}_median'] = np.median(rmsf_values)
        result[f'{cdr}_max'] = np.max(rmsf_values)
        result[f'{cdr}_min'] = np.min(rmsf_values)

    return result


def main(inpath, n_jobs=1):
    dirs = sorted([d for d in glob(f'{inpath}/*') if os.path.isdir(d)])

    results = []
    if n_jobs == 1:
        for d in tqdm(dirs, desc='Processing directories', total=len(dirs)):
            result = process_directory(d)
            if result is not None:
                results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for result in tqdm(executor.map(process_directory, dirs), desc='Processing directories', total=len(dirs)):
                if result is not None:
                    results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    main(args.pred_path, n_jobs=args.n_jobs)