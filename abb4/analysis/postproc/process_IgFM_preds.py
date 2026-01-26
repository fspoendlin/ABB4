'''Script to reformat IgFM predictions to be compatible with the rest of the pipeline'''

import pandas as pd
import numpy as np
from glob import glob
import os
import pickle
from tqdm import tqdm
import shutil
import argparse
from joblib import Parallel, delayed
from helper_functions.process_pdbs import apply_to_pdb_file, rename_2nd_chain
from helper_functions.renumber_pdbs import renumber_pdb
from helper_functions.sequence import long_2_short
from anarci import run_anarci

path = '/ceph/opig-shared/projects/spoendlin-EnsemblePrediction/EnsembleStructures/IgFM_predictions/'
# path = '../IgFM_predictions/'

def get_seperator_idx(seq):
    result = run_anarci(seq)[2][0][0]
    if result['query_start'] > 20: # second chain recognized
        sep = result['query_start']
    else: # first chain recognized
        sep = result['query_end']
    return sep

def sep_index_for_pdb(mab_row):
    seq = mab_row.aa_seq.values[0].replace('X', '')
    return get_seperator_idx(seq)

def process_pdbs_IgFM(infile, outfile, mab_row):
    id = sep_index_for_pdb(mab_row)
    apply_to_pdb_file(infile, outfile, rename_2nd_chain, original_chain='H', new_chain_label='L', second_chain_idx=id)
    # apply_to_pdb_file(out_path, out_path, rename_chains, new_label='H', old_label='A')
    renumber_pdb(outfile, outfile)


def main_single(prediction, epoch, steps, i, dataset_path):
    dataset = pd.read_csv(dataset_path)

    training_path = glob(f'../../../IgFM/ckpt/ab_folding/fwd_fold_mAbs_{prediction}/*')[0]
    test_struc_path = f'{training_path}/test/'
    processed_path = f'{path}/{prediction}_{epoch}_{steps}/predictions/'

    dirs = [d for d in glob(os.path.join(test_struc_path, '*/')) if os.path.isdir(d)]

    print(f'Processing {len(dirs)} directories')
    for dir in tqdm(dirs):

        mab_id = dir.split('/')[-2].split('_')[1]
        mab = dataset[dataset.pdb_name_enc == int(mab_id)]
        pdb_name = mab.pdb_name.values[0]

        pred_struc = os.path.join(dir, f'pred_struc_epoch_{epoch}_steps_{steps}_i_{i}_1.pdb')
        dest_dir = os.path.join(processed_path, pdb_name)
        os.makedirs(dest_dir, exist_ok=True)
        dest_file = os.path.join(processed_path, pdb_name, f'{pdb_name}_model_000.pdb')

        process_pdbs_IgFM(pred_struc, dest_file, mab)

def reformat_pickle(infile, outfile):
    with open(infile, 'rb') as f:
        data = pickle.load(f)

    data_new = {}
    data_new['atom_positions'] = data['pred_atom37']
    data_new['bb_mask'] = data['res_mask'].astype(int)
    data_new['residue_index'] = data['imgt']
    data_new['aatype']  = data['aatypes']

    with open(outfile, 'wb') as f:
        pickle.dump(data_new, f)

    # compute dc
    CA_index = 1
    HC_mask = data_new['residue_index'] < 1000
    LC_mask = data_new['residue_index'] >= 1000
    HC_mean = np.mean(data_new['atom_positions'][:, CA_index, :][HC_mask], axis=0)
    LC_mean = np.mean(data_new['atom_positions'][:, CA_index, :][LC_mask], axis=0)
    dc = np.linalg.norm(HC_mean - LC_mean)

    return dc

def process_ensemble(pred_dir, pred_path):
    name = pred_dir.split('/')[-1]        
    outdir  = f'{pred_path}/predictions_processed/{name}'
    os.makedirs(outdir, exist_ok=True)

    files = glob(f'{pred_dir}/sample_*.pkl')
    assert len(files) == 100
    metadata = []
    for file in files:
        i = int(file.split('_')[-1].split('.')[0])
        outfile = f'{outdir}/{name}_model_{i:03d}.pkl'
        dc = reformat_pickle(file, outfile)

        metadata.append({
            'structure': f'{name}_model_{i:03d}',
            'pdb_name': name,
            'model_id': i,
            'processed_path': outfile,
            'raw_path': file,
            'dc': dc
        })

    return metadata

def process_esemble_pdbs(pred_dir, pred_path, dataset):
    try:
        name = pred_dir.split('/')[-1]
        outdir  = f'{pred_path}/predictions/{name}'
        os.makedirs(outdir, exist_ok=True)
        files = glob(f'{pred_dir}/sample_*.pdb')
        assert len(files) == 100

        for file in files:
            pdb_name = file.split('/')[-2]
            model_idx = int(file.split('_')[-1].split('.')[0])
            if 'ABB3_set_map' in dataset.columns:
                mab_row = dataset[dataset.ABB3_set_map == pdb_name]
            else:
                mab_row = dataset[dataset.pdb_name == pdb_name]
            assert mab_row.shape[0] >= 1
            mab_row = mab_row.sample(1)

            out_file = os.path.join(outdir, f'{pdb_name}_model_{model_idx:03d}.pdb')
            process_pdbs_IgFM(file, out_file, mab_row)
    except AssertionError:
        print(f'{pred_dir} not found in dataset')
        # raise AssertionError

def ensemble_pdbs(pred_path, dataset_path, n_jobs=-1):
    dataset = pd.read_csv(dataset_path) 

    pred_dirs = [d for d in glob(f'{pred_path}/predictions_raw/*') if os.path.isdir(d)]

    print(f'Processing {len(pred_dirs)} directories')
    _ = Parallel(n_jobs=n_jobs)(
            delayed(process_esemble_pdbs)(
                pred_dir, pred_path, dataset
                ) for pred_dir in tqdm(pred_dirs)
    )

def main_ensemble(pred_path, dataset_path, process_pickles=True, process_pdbs=False, n_jobs=-1):

    if process_pickles:
        pred_dirs = [d for d in glob(f'{pred_path}/predictions_raw/*') if os.path.isdir(d)]

        print(f'Processing {len(pred_dirs)} directories')
        metadata = Parallel(n_jobs=n_jobs)(delayed(process_ensemble)(pred_dir, pred_path) for pred_dir in tqdm(pred_dirs))
        metadata = [item for sublist in metadata for item in sublist]
        metadata = pd.DataFrame(metadata)
        metadata.to_csv(f'{pred_path}/predictions_processed/metadata.csv', index=True)

    if process_pdbs:
        print('Processing PDB files')
        ensemble_pdbs(pred_path, dataset_path, n_jobs=n_jobs)


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--pred_path', type=str, required=True, help='Path to modelpredictions')
    argparse.add_argument('--prediction', type=str, help='Prediction name')
    argparse.add_argument('--epoch', type=str, help='Epoch number')
    argparse.add_argument('--i', type=str, default='0', help='i')
    argparse.add_argument('--pred_type', type=str, default='single', help='Output directory')
    argparse.add_argument('--n_jobs', type=int, default=30, help='Number of jobs for ensemble prediction')
    argparse.add_argument('--process_pickles', action='store_true', help='Process pickles for ensemble prediction')
    argparse.add_argument('--process_pdbs', action='store_true', help='Process PDBs for ensemble prediction')
    argparse.add_argument('--dataset_path', type=str, default='/ceph/opig-shared/users/spoendli/conformation_prediction/IgFM/metadata/ABB3_latest_CDRsim_metadata_random_ceph_paths.csv')

    args = argparse.parse_args()
    if args.pred_type == 'single': # single structure prediction
        main_single(args.prediction, args.epoch, args.steps, args.i, args.dataset_path)
    elif args.pred_type == 'ensemble': # ensemble prediction
        main_ensemble(
            pred_path=args.pred_path,
            dataset_path=args.dataset_path,
            process_pdbs=args.process_pdbs,
            process_pickles=args.process_pickles,
            n_jobs=args.n_jobs
        )

    
    print('Finished processing predictions')

