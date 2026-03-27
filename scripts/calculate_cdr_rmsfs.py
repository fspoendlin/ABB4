"""Calculate per-residue CDR RMSF statistics across predicted antibody structure ensembles.

For each per-target subdirectory under ``--pred_path``, loads all ``.pdb``
files as a trajectory using MDAnalysis, aligns frames to the first structure on
Cα atoms, and computes per-residue RMSF values for each CDR loop
(CDRH1/2/3, CDRL1/2/3). Aggregated statistics (mean, std, median, max, min)
per CDR are written incrementally to ``<pred_path>/cdr_rmsfs.csv``.

IMGT residue ranges used (heavy and light chains share the same numbering):

    CDR1: 27–38   CDR2: 56–65   CDR3: 105–117

Usage
-----
    python scripts/calculate_cdr_rmsfs.py \\
        --pred_path predictions_imgt/

Arguments
---------
--pred_path : str
    Root directory containing one subdirectory per target, each holding
    IMGT-renumbered ``sample_*.pdb`` files.

Output
------
``<pred_path>/cdr_rmsfs.csv``
    CSV with one row per target and columns for each CDR loop statistic.
"""

import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
from glob import glob
from MDAnalysis.analysis import align
import argparse
import os
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', type=str, required=True, help='Path to directories with PDB files')
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


def main(inpath):
    dirs = [d for d in glob(f'{inpath}/*') if os.path.isdir(d)]

    results = []
    for d in tqdm(dirs, desc='Processing directories', total=len(dirs)):
        code = os.path.basename(d)
        pdb_files = glob(f'{d}/*.pdb')

        if len(pdb_files) <= 1:
            continue

        result = {'code': code}
        for cdr in ['CDRH1', 'CDRH2', 'CDRH3', 'CDRL1', 'CDRL2', 'CDRL3']:
            chain = cdr[3]
            rmsf_values = calculate_CDR_rmsf(pdb_files, chain, cdr[:-2]+cdr[-1])
            result[f'{cdr}_mean'] = np.mean(rmsf_values)
            result[f'{cdr}_std'] = np.std(rmsf_values)
            result[f'{cdr}_median'] = np.median(rmsf_values)
            result[f'{cdr}_max'] = np.max(rmsf_values)
            result[f'{cdr}_min'] = np.min(rmsf_values)
        results.append(result)
        
        df = pd.DataFrame(results)
        df.to_csv(f'{inpath}/cdr_rmsfs.csv', index=False)


if __name__ == '__main__':
    main(args.pred_path)
