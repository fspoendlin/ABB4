from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import warnings
import argparse
import os

def safe_pickle_dump(obj, path):
    tmp_path = path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)  # atomic replace

def add_rmsd_rmsf_2_pkl(row, ensemble_metrics, dset):
    '''Add RMSD and RMSF data to the pkl data from ensemble metrics.

    Will add the following keys to the pkl data:
    - rmsd: numpy array of shape (9, 3) with RMSD values ordered as global, HC, LC, CDRH1, CDRH2, CDRH3, CDRL1, CDRL2, CDRL3
    - rmsf: numpy array of shape (3, n_residues) with RMSF values ordered as global_rmsf, global_rmsf_std, chain_rmsf
    '''
    # try:
    pkl_path = row['processed_path']
    # pkl_path = pkl_path.replace('/ceph/opig-shared/', '/vols/opig/')
    if dset == 'aa_sim':
        h_chain = 'H'
        l_chain = 'L'
    else:
        h_chain = row.simID[-2]
        l_chain = row.simID[-1]

    with open(pkl_path, 'rb') as f:
        warnings.filterwarnings(
            "ignore",
            message=r".*numpy\.core\.numeric.*",
            category=DeprecationWarning,
        )
        pkl_data = pickle.load(f)

    rmsd_rmsf = ensemble_metrics.loc[row.simID]

    # load rmsd data
    rmsd = []
    for name in ['global', 'HC', 'LC', 'CDRH1', 'CDRH2', 'CDRH3', 'CDRL1', 'CDRL2', 'CDRL3']:
        rmsd.append(np.array(rmsd_rmsf[name]))
    rmsd = np.array(rmsd)
    assert rmsd.shape == (9, 3), f'RMSD shape mismatch: {rmsd.shape}'


    # load rmsf data
    # prepare residue mask
    hc_mask = (rmsd_rmsf['chain'] == h_chain)
    lc_mask = (rmsd_rmsf['chain'] == l_chain)
    imgt_numeric = np.array([int(x.split('.')[0]) for x in rmsd_rmsf['imgt']])
    res_mask = imgt_numeric <= 128
    assert (imgt_numeric[res_mask].shape[0] == pkl_data['residue_index'].shape[0])          # same total length
    hc_res_index = pkl_data['residue_index'][pkl_data['residue_index'] <= 128]
    lc_res_index = pkl_data['residue_index'][pkl_data['residue_index'] > 128] - 1000
    assert (imgt_numeric[res_mask][hc_mask[res_mask]].shape[0] == hc_res_index.shape[0])    # same hc length
    assert (imgt_numeric[res_mask][hc_mask[res_mask]][0] == hc_res_index[0])                # same hc first residue
    assert (imgt_numeric[res_mask][hc_mask[res_mask]][-1] == hc_res_index[-1])              # same hc last residue
    assert (imgt_numeric[res_mask][lc_mask[res_mask]].shape[0] == lc_res_index.shape[0])    # same lc length
    assert (imgt_numeric[res_mask][lc_mask[res_mask]][0] == lc_res_index[0])                # same lc first residue
    assert (imgt_numeric[res_mask][lc_mask[res_mask]][-1] == lc_res_index[-1])              # same lc last residue

    # hc_mask = (rmsd_rmsf['chain'] == h_chain)
    # lc_mask = (rmsd_rmsf['chain'] == l_chain)
    # imgt_numeric = np.array([int(x.split('.')[0]) for x in rmsd_rmsf['imgt']])
    # res_mask = imgt_numeric <= 128
    # imgt_numeric = imgt_numeric + lc_mask.astype(int) * 1000
    # assert (imgt_numeric[res_mask] == pkl_data['residue_index']).all()

    # rmsf
    rmsf = []
    for name in ['global_rmsf', 'global_rmsf_std', 'chain_rmsf']:
        rmsf.append(np.array(rmsd_rmsf[name])[res_mask])
    rmsf = np.array(rmsf)
    assert rmsf.shape == (3, len(pkl_data['residue_index'])), f'RMSF shape mismatch: {rmsf.shape}'

    pkl_data['rmsd'] = rmsd
    pkl_data['rmsf'] = rmsf

    warnings.filterwarnings(
            "ignore",
            message=r".*numpy\.core\.numeric.*",
            category=DeprecationWarning,
        )
    safe_pickle_dump(pkl_data, pkl_path)

    return f'Success, {row.pdb_name}'
    
    # except Exception as e:
    #     return f'Error, {row.pdb_name}, {e}'




def main(ensemble_metrics_file, pred_dir, dset, batch, n_jobs=30):
    # load ensemble metrics
    with open(ensemble_metrics_file, 'rb') as f:
        ensemble_metrics = pickle.load(f)
    ensemble_metrics = pd.DataFrame(ensemble_metrics).T

    # load metadata file
    metadata = pd.read_csv(pred_dir+'/metadata.csv')
    if dset == 'original':
        metadata['simID'] = metadata['pdb_name'].apply(lambda x: x.split('frame')[0][:-1])
    elif dset == 'random':
        metadata['simID'] = metadata['pdb_name'].apply(lambda x: x[:-5])
    elif dset == 'aa_sim':
        metadata['simID'] = metadata['simulation_ID']


    # start = batch * 1000
    # end = (batch + 1) * 1000
    # metadata = metadata.iloc[start:end]

    # # parallel processing
    # messages = Parallel(n_jobs=n_jobs)(delayed(add_rmsd_rmsf_2_pkl)(row, ensemble_metrics) for _, row in tqdm(metadata.iterrows(), total=metadata.shape[0], desc='Processing pkl files'))

    messages = []
    for _, row in tqdm(metadata.iterrows(), total=metadata.shape[0], desc='Processing pkl files'):
        try:
            # print(f'Processing {row.simID}...')
            messages.append(add_rmsd_rmsf_2_pkl(row, ensemble_metrics, dset))
        except Exception as e:
            print(f'Error processing {row.pdb_name}: {e}')
            continue

    with open(f'error_log_{batch}.pkl', 'wb') as f:
        pickle.dump(messages, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble_metrics', type=str, required=True)
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--dset', type=str, choices=['original', 'random', 'aa_sim'], default='original')
    parser.add_argument('--n_jobs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=0)
    args = parser.parse_args()

    main(args.ensemble_metrics, args.pred_dir, args.dset, args.batch, args.n_jobs)

# cd IgFM/multiflow/data/preproc/
# conda activate ig_flow
# python add_ensemble_metrics.py --ensemble_metrics /vols/opig/projects/spoendlin-EnsemblePrediction/CDRsim_abb2_random_frames/CDRsim_ensmble_metrics.pkl --pred_dir /vols/opig/projects/spoendlin-EnsemblePrediction/CDRsim_abb2_random_frames/processed/ --dset random --batch 0
