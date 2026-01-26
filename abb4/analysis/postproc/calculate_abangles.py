
import argparse
from ABDB.AbPDB import AntibodyParser
from ABDB.ABangle import abangle
import os
from glob import glob
import pandas as pd
from joblib import delayed, Parallel


ab_parser = AntibodyParser(scheme='imgt', definition='imgt')
angles_parser = abangle()

def get_angles_for_name(name, pred_dir, model):
    pdbs = glob(pred_dir + f'/predictions/{name}/*.pdb')
    ab_angles = []

    try:
        assert len(pdbs) == 100
    except AssertionError as e:
        if model == 'bioemu':
            assert len(pdbs) == 99
        else:
            raise AssertionError

    for pdb in pdbs:
        model_id = os.path.basename(pdb)[-7:-4]
        ab_structure = ab_parser.get_antibody_structure(pdb, pdb)
        ab_angle = angles_parser.calculate_angles(ab_structure)
        ab_angle['name'] = name
        ab_angle['model_id'] = model_id
        ab_angles.append(ab_angle)
    return ab_angles

def get_angles_for_gt(row):
    pdb = row.raw_path
    name = row.ABB3_set_map
    model_id = row.pdb_name.split('frame')[1][1:]

    ab_structure = ab_parser.get_antibody_structure(pdb, pdb)
    ab_angle = angles_parser.calculate_angles(ab_structure)
    ab_angle['name'] = name
    ab_angle['model_id'] = model_id
    return ab_angle

def main(model, dset, n_jobs):
    pred_dir = f'/vols/opig/users/spoendli/conformation_prediction/conformation_prediction/analysis/conformation_prediction_benchmark/{model}'
    if os.path.exists(pred_dir) is False:
        pred_dir = f'/vols/opig/users/spoendli/conformation_prediction/conformation_prediction/analysis/IgFM_predictions/{model}'

    dataset = pd.read_csv(
        '/vols/opig/users/spoendli/conformation_prediction/IgFM/metadata/ABB3_latest_metadata.csv',
        index_col=0, low_memory=False
    )
    dataset = dataset[dataset.latest_split == dset]
    names = dataset.pdb_name.values


    ab_angles = Parallel(n_jobs=n_jobs)(
        delayed(get_angles_for_name)(name, pred_dir, model) for name in names
    )
    ab_angles = sum(ab_angles, [])
    try:
        assert len(ab_angles) == 100 * len(names)
    except AssertionError as e:
        if model == 'bioemu':
            pass
        else:
            raise AssertionError

    ab_angles = pd.DataFrame(ab_angles)
    ab_angles.to_csv(f'{pred_dir}/{dset}_ab_angles.csv')

def main_gt(dset, n_jobs):
    dataset = pd.read_csv(
        '/vols/opig/users/spoendli/conformation_prediction/IgFM/metadata/ABB3_latest_CDRsim_metadata.csv',
        index_col=0, low_memory=False
    )
    dataset = dataset[dataset.latest_split == 'val']
    ab_angles = Parallel(n_jobs=30)(
        delayed(get_angles_for_gt)(row) for 
        i, row in dataset.iterrows()
    )
    ab_angles = pd.DataFrame(ab_angles)
    ab_angles.to_csv(f'../CDRsim_{dset}_ab_angles.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dset', type=str, required=True)
    parser.add_argument('--n_jobs', type=int, default=30)
    args = parser.parse_args()

    if args.model == 'gt':
        main_gt(args.dset, args.n_jobs)
    else:
        main(args.model, args.dset, args.n_jobs)
