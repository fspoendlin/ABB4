#!/usr/bin/env python3
"""Produce long-format per-residue CDR RMSF CSV without modifying the original script.

Output columns: code, chain, cdr, resid, resname, rmsf, frame_count
"""
import argparse
import os
from glob import glob
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

import MDAnalysis as mda
from MDAnalysis.analysis import rms, align


cdr2range = {
    'CDR1': (27, 38),
    'CDR2': (56, 65),
    'CDR3': (105, 117),
}


def calculate_CDR_rmsf_per_residue(pdb_files, chain, cdr):
    u = mda.Universe(pdb_files[0], pdb_files)
    reference = mda.Universe(pdb_files[0])

    aligner = align.AlignTraj(
        u,
        reference,
        select=f"chainID {chain} and name CA",
        in_memory=True,
    )
    aligner.run()

    sel = u.select_atoms(f"chainID {chain} and name CA and resid {cdr2range[cdr][0]}-{cdr2range[cdr][1]}")
    if len(sel) == 0:
        return []

    rmsf = rms.RMSF(sel).run()
    rmsf_values = rmsf.results.rmsf

    # Use the reference to extract residue numbering and names (consistent across dirs)
    ref_sel = reference.select_atoms(f"chainID {chain} and name CA and resid {cdr2range[cdr][0]}-{cdr2range[cdr][1]}")
    # ref_sel should contain one CA atom per residue; map atoms -> residues
    residues = [atom.residue for atom in ref_sel.atoms]

    # If lengths mismatch, fall back to resid-based residues
    if len(residues) != len(rmsf_values):
        residues = list(ref_sel.residues)

    frame_count = len(pdb_files)

    rows = []
    for res, rmsf_val in zip(residues, rmsf_values):
        rows.append({
            'resid': int(res.resid),
            'resname': res.resname,
            'rmsf': float(rmsf_val),
            'frame_count': int(frame_count),
        })

    return rows


def process_directory(d):
    code = os.path.basename(d)
    pdb_files = sorted(glob(f'{d}/*.pdb'))
    if len(pdb_files) == 0:
        return None

    all_rows = []
    for cdr_full in ['CDRH1', 'CDRH2', 'CDRH3', 'CDRL1', 'CDRL2', 'CDRL3']:
        chain = cdr_full[3]
        cdr_key = cdr_full[:-2] + cdr_full[-1]
        try:
            per_res_rows = calculate_CDR_rmsf_per_residue(pdb_files, chain, cdr_key)
        except Exception as e:
            print(f"Error processing {code} {cdr_full}: {e}")
            continue

        for r in per_res_rows:
            r.update({'code': code, 'chain': chain, 'cdr': cdr_key})
            all_rows.append(r)

    return all_rows


def main(inpath, n_jobs=1, output_long_csv='cdr_rmsf_long.csv'):
    dirs = sorted([d for d in glob(f'{inpath}/*') if os.path.isdir(d)])

    results = []
    if n_jobs == 1:
        for d in tqdm(dirs, desc='Processing directories', total=len(dirs)):
            result = process_directory(d)
            if result:
                results.extend(result)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for result in tqdm(executor.map(process_directory, dirs), desc='Processing directories', total=len(dirs)):
                if result:
                    results.extend(result)

    if len(results) == 0:
        print('No results to write.')
        return

    df = pd.DataFrame(results)
    df = df.sort_values(['code', 'cdr', 'resid']).reset_index(drop=True)
    df.to_csv(output_long_csv, index=False)
    print(f'Wrote {len(df)} rows to {output_long_csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate per-residue CDR RMSF and write long-format CSV')
    parser.add_argument('--pred_path', type=str, required=True, help='Path to directories with PDB files')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of processes for directory-level parallelism')
    parser.add_argument('--output_long_csv', type=str, default='cdr_rmsf_long.csv', help='Output CSV file name for the long-format results')
    args = parser.parse_args()

    main(args.pred_path, n_jobs=args.n_jobs, output_long_csv=args.output_long_csv)
