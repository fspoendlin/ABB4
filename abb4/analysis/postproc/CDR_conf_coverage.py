import pandas as pd
import MDAnalysis as mda
import pandas as pd
import numpy as np
from glob import glob
from MDAnalysis.analysis import align, rms
from tqdm import tqdm
tqdm.pandas()
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from helper_functions.reg_def_mAbs import reg_def
from joblib import Parallel, delayed
import argparse
import os
from joblib import parallel_backend


cdr2imgt = {
    'CDRH1': (27, 38),
    'CDRH2': (56, 65),
    'CDRH3': (105, 117),
    'CDRL1': (27, 38),
    'CDRL2': (56, 65),
    'CDRL3': (105, 117)
}

chain2idx = {
    'H': 0,
    'L': 1
}

fw = reg_def['fwH']  # labeled as H but this is framework in chain numbered from 0

def load_predicted_sturcs(row, pred_path):
    pdb_name = row.select
    files = glob(f'{pred_path}/{pdb_name}/*.pdb')
    try:
        assert len(files) == 100
    except AssertionError: # few exceptions with 99 strucs for bioemu
        print(f'{pdb_name}, {len(files)} number of files found')
    return mda.Universe(files[0], files)

def load_xray_strucs(row):
    xray_us = []
    seen_confs = []
    for i, file in enumerate(row.paths):
        p = f"{file}"
        xray_us.append(mda.Universe(p, p))
        seen_confs.append(row['conformation_labels'][i])
    return xray_us, seen_confs

def rmsd_mat(xray_us, preds_u, xray_chains, pred_chain, imgt_start, imgt_end, align_resi, asrt_res_id):
    rmsds = np.full((len(xray_us), 100), np.nan)
    for i, xray_u in enumerate(xray_us):
        # align preds to xray
        if asrt_res_id:
            assert (
                (xray_u.select_atoms(f'chainID {pred_chain} and resid {" ".join([str(x) for x in align_resi])} and name CA').residues.resnames ==
                preds_u.select_atoms(f'chainID {pred_chain} and resid {" ".join([str(x) for x in align_resi])} and name CA').residues.resnames).all()
            )
        else:
            pass
        align.AlignTraj(
            preds_u,
            xray_u,
            select=f'chainID {pred_chain} and resid {" ".join([str(x) for x in align_resi])} and name CA',
            in_memory=True
        ).run()

        xray_sele = xray_u.select_atoms(
            f'chainID {pred_chain} and resid {imgt_start}-{imgt_end} and name CA'
        )
        if xray_sele.atoms.resids.shape[0] == 0: # formatting of PDB file can change MDA residue numbering
            xray_sele = xray_u.select_atoms(
                f'chainID {pred_chain} and resid 1{imgt_start}-1{imgt_end} and name CA'
            )
        if xray_sele.atoms.resids.shape[0] == 0:
            xray_sele = xray_u.select_atoms(
                f'chainID {pred_chain} and resid 10{imgt_start}-10{imgt_end} and name CA'
            )
        
        preds_u_sele = preds_u.select_atoms(
            f'chainID {pred_chain} and resid {imgt_start}-{imgt_end} and name CA'
        )
        assert preds_u_sele.residues.resnames.shape[0] > 0, 'No residues found'
        assert (xray_sele.residues.resnames == preds_u_sele.residues.resnames).all(), 'Residue names do not match'

        xray_sele_coords = xray_sele.atoms.positions
        for j, ts in enumerate(preds_u.trajectory):
            preds_u_coords = preds_u_sele.atoms.positions
            rmsd = rms.rmsd(xray_sele_coords, preds_u_coords, superposition=False)
            rmsds[i, j] = rmsd
        
    return rmsds

def get_align_residues(name):
    # hard coded exception due to missing residues
    if name == '6mtn_HL':
        return fw[5:]
    elif name == '4jy5_HL':
        return fw[4:]
    else:
        return fw
    
def assert_residue_identity(name, chain):
    # hard coded exception due to insetions
    if name == '6umi_HL' and chain == 'H':
        return False
    else:
        return True

def exp_ensembles_precision_recall(row, pred_path, CDR):
    # load strucs
    preds_u = load_predicted_sturcs(row, pred_path)
    xray_us, seen_confs = load_xray_strucs(row)

    # define params
    pred_chain = CDR[3].upper()
    chain_idx = chain2idx[pred_chain]
    xray_chains = [x[1][chain_idx] for x in row.pdbs_chains]
    imgt_start, imgt_end = cdr2imgt[CDR]
    align_resi = get_align_residues(row.select)
    asrt_res_id = assert_residue_identity(row.select, pred_chain)

    # calculate rmsd matrix
    rmsds = rmsd_mat(
        xray_us,
        preds_u,
        xray_chains,
        pred_chain,
        imgt_start,
        imgt_end,
        align_resi,
        asrt_res_id
    )

    # calculate metrics
    precision = rmsds.min(axis=0)

    recall = np.nanmin(rmsds, axis=1)
    recall_new = np.full(len(np.unique(seen_confs)), np.nan)
    for i, j in enumerate(set(seen_confs)):
        indices = np.where(np.array(seen_confs) == j)
        recall_new[i] = recall[indices].mean()

    return precision, recall_new
    # return precision, np.array([recall_new.mean(), np.median(recall_new), recall_new.min(), recall_new.max()])

def process_single_row(row, pred_path, CDR):
    """Process a single row for precision/recall calculation"""
    try:
        result = exp_ensembles_precision_recall(row, pred_path, CDR)
        return result  # Return index and result
    except Exception as e:
        print(f"Error processing {row.select}: {e}")
        return np.full(100, np.nan), np.full(4, np.nan)

def process_CDR(CDR, pred_path, RMSD_threshold='lower', n_jobs=10):
    # preprocess datafile
    df = pd.read_pickle(
        f'/ceph/opig-shared/projects/cagiada-CDRsimulations/analysis_files/CDRsimulations/data/crystal/confs_data/IMGT_loop_defs/{CDR}_Fv_flexibility_align_Fv_v3.pkl'
    )
    df["select"] = df["pdbs_chains"].apply(lambda x: x[0] if isinstance(x, list) and x else x)
    df['select'] = df['select'].apply(lambda x: f'{x[0]}_{x[1]}')
    path_prefix = '/ceph/opig-shared/projects/spoendlin-EnsemblePrediction/exp_val/pdb_new/'
    df['paths'] = df['paths'].apply(lambda xs: [f'{path_prefix}{os.path.basename(x)}' for x in xs])

    if RMSD_threshold == 'lower':
        df['conformation_labels'] = df['labels_.5A']
    elif RMSD_threshold == 'higher':
        df['conformation_labels'] = df['labels_1.25A']
    else:
        raise ValueError("RMSD_threshold must be 'lower' or 'higher'")

    # Parallelize the processing
    with parallel_backend("loky"):
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_row)(row, pred_path, CDR) 
            for i, row in tqdm(df.iterrows(), total=len(df), desc=f'Processing {CDR}')
        )
    
    # save results
    precision = [result[0] for result in results]
    precision = pd.DataFrame(precision, index=df.select.values)
    precision.to_csv(f'{str(Path(pred_path).parent)}/{CDR}_precision_{RMSD_threshold}.csv')

    recall = [result[1] for result in results]
    recall = pd.DataFrame(recall, index=df.select.values)
    recall.to_csv(f'{str(Path(pred_path).parent)}/{CDR}_recall_{RMSD_threshold}.csv')


def main(method, RMSD_threshold, pred_path, n_jobs):
    path = f'{pred_path}/{method}/predictions/'
    for CDR in ['CDRH1', 'CDRH2', 'CDRH3', 'CDRL1', 'CDRL2', 'CDRL3']:
        process_CDR(CDR, path, RMSD_threshold=RMSD_threshold, n_jobs=n_jobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate conformation coverage.')
    parser.add_argument('--method', type=str, required=True, help='Method to evaluate')
    parser.add_argument('--RMSD_threshold', type=str, default='lower', help='RMSD threshold for evaluation (lower=0.5A, higher=1.25A)')
    parser.add_argument('--pred_path', type=str, default='/ceph/opig-shared/projects/spoendlin-EnsemblePrediction/exp_val/predictions/', help='Path to predictions')
    parser.add_argument('--n_jobs', type=int, default=30, help='Number of parallel jobs to run.')

    args = parser.parse_args()
    main(args.method, args.RMSD_threshold, args.pred_path, n_jobs=args.n_jobs)
