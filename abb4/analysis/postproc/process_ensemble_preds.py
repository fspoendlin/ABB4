import argparse
import dataclasses
import functools as fn
import pandas as pd
import os, warnings
import multiprocessing as mp
import time
from Bio import PDB
from Bio.PDB import Chain, Residue
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import numpy as np
import mdtraj as md
import sys
from glob import glob


from abb4.data import utils as du
from abb4.data import parsers
from abb4.data import errors
from abb4.data import residue_constants


# Define the parser
parser = argparse.ArgumentParser(
    description='PDB processing script.')
parser.add_argument(
    '--pdb_dir',
    help='Path to directory with PDB files.',
    type=str)
parser.add_argument(
    '--num_processes',
    help='Number of processes.',
    type=int,
    default=40)
parser.add_argument(
    '--write_dir',
    help='Path to write results to.',
    type=str,
    default='preprocessed')
parser.add_argument(
    '--verbose',
    help='Whether to log everything.',
    action='store_true')


def process_file(file_path: str, write_dir: str):
    """
    Processes antibody PDB file into usable, smaller pickles.
    Notably combines heavy and light chains into a single chain called 'M'
    (merged).

    Args:
        file_path: Path to file to read.
        write_dir: Directory to write pickles to.

    Returns:
        Saves extracted protein features to pickled dict and returns metadata.
        Pickle contains the following keys:
            - atom37: np.ndarray of shape (N, 37, 3) with atom positions.
            - aatype: np.ndarray of shape (N,) with amino acid types for both 
                chains (numeric; heavy chain first).
            - atom_mask: np.ndarray of shape (N, 37) with atom masks.
            - imgt: np.ndarray of shape (N,) with numeric part of imgt numbering 
                (heavy chain first, light chain index skips to 1001).
            - chain_index: np.ndarray of shape (N,) with chain indices.
                (should contain only '38' repeated N times, the encoding of 'M',
                as per data.utils.chain_str_to_int())
            - b_factors: np.ndarray of shape (N, 37) with B-factors.
            - res_mask: np.ndarray of shape (N,) with resiude masks. 
                (whether to inlcude the C-alpha of this residue)
            - dc: float, distance between heavy and light chain center of mass
        Metadata contains the following columns:
            - structure: name of the PDB file
            - pdb_name: PDB ID
            - model_id: model index
            - processed_path: path to the pickled file
            - raw_path: path to the raw PDB file
            - num_chains: number of chains in the unprocessed PDB file (should be 2)
            - seq_len: total sequence length of the complex (heavy+light chain)
                (we enforce agreement with seqlen in input csv)
            - modeled_seq_len: sequence length of the complex with only modeled residues
                (should hopefully agree with seq_len)
            - dc: distance between heavy and light chain center of mass

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propagated.
    """
    metadata = {}
    file_name = os.path.basename(file_path).replace('.pdb', '')
    metadata['structure'] = file_name
    metadata['pdb_name'] = f'{file_name.split("_")[0]}_{file_name.split("_")[1]}'
    metadata['model_id'] = int(file_name.split("_")[-1])

    processed_dir = os.path.join(write_dir, metadata['pdb_name'])
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, f'{file_name}.pkl')
    metadata['processed_path'] = os.path.abspath(processed_path)
    metadata['raw_path'] = file_path

    with warnings.catch_warnings():

        # load structure
        warnings.simplefilter('ignore', PDBConstructionWarning)
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure(file_name, file_path)

        # count chains
        struct_chains = {chain.id.upper() : chain for chain in structure.get_chains()}
        num_chains = len(struct_chains)
        if num_chains != 2:
            raise errors.DataError(f'Expected 2 chains, got {num_chains} in file {file_path}.')
        metadata['num_chains'] = num_chains
        
        # ensure chains have correct IDs
        if 'H' not in struct_chains or 'L' not in struct_chains:
            raise errors.DataError(f'Expected chains H and L, got {list(struct_chains.keys())} in file {file_path}.')
        
        # get amino acid sequences
        aa_seq = ''
        for chain in structure.get_chains():
            for res in chain:
                aa_seq += residue_constants.restype_3to1.get(res.resname, 'X')
        metadata['aa_seq'] = aa_seq

        # offset light-chain indices by 1000                                                # NOTE: index definition
        modified_light_chain = Chain.Chain("L")
        offset = 1000
        for res in struct_chains['L']:
            new_id = (res.id[0], res.id[1] + offset, res.id[2])
            modified_res = Residue.Residue(new_id, res.resname, res.segid)
            for atom in res:
                modified_res.add(atom)
            modified_light_chain.add(modified_res)
        
        # merge modified light chain and heavy chain into a single chain
        merged_chain = Chain.Chain("M")
        for chain in [struct_chains['H'], modified_light_chain]:
            for res in chain:
                merged_chain.add(res)
        struct_chains = {"M": merged_chain}
    
        # Extract Features
        # bb_positions, atom_positions, masks
        struct_feats = []
        all_seqs = set()
        for chain_id, chain in struct_chains.items():
            chain_id = du.chain_str_to_int(chain_id) # ascii index of 'M'
            chain_prot = parsers.process_chain(chain, chain_id)                             # NOTE: index definition
            chain_dict = dataclasses.asdict(chain_prot)
            chain_dict = du.parse_chain_feats(chain_dict)
            all_seqs.add(tuple(chain_dict['aatype']))
            struct_feats.append(chain_dict)
        if len(all_seqs) != 1:
            raise errors.DataError(f'Failed to combine chains for file {file_path}.')
        complex_feats = du.concat_np_features(struct_feats, False)

        # aa sequence and sequence length
        complex_aatype = complex_feats['aatype']
        metadata['seq_len'] = len(complex_aatype)
        modeled_idx = np.where(complex_aatype != 20)[0] # no unnatural AAs
        if np.sum(complex_aatype != 20) == 0:  # all-unnatural AAs
            raise errors.LengthError('Protein contains no residues that are modelled (natural AAs).')
        min_modeled_idx, max_modeled_idx = np.min(modeled_idx), np.max(modeled_idx)
        metadata['modeled_seq_len'] = max_modeled_idx - min_modeled_idx + 1                 # NOTE: index definition
        complex_feats['modeled_idx'] = modeled_idx                                          # NOTE: index definition

        # VH-VL distance
        CA_index = residue_constants.atom_order['CA']
        HC_mask = complex_feats['residue_index'] < 1000
        LC_mask = complex_feats['residue_index'] >= 1000
        HC_mean = np.mean(complex_feats['atom_positions'][:, CA_index, :][HC_mask], axis=0)
        LC_mean = np.mean(complex_feats['atom_positions'][:, CA_index, :][LC_mask], axis=0)
        dc = np.linalg.norm(HC_mean - LC_mean)
        metadata['dc'] = dc
        complex_feats['dc'] = dc

        # Write features to pickles.
        du.write_pkl(processed_path, complex_feats)

        # Return metadata
        return metadata
    
def process_serially(all_paths, write_dir):
    all_metadata = []
    for i, file_path in enumerate(all_paths):
        try:
            start_time = time.time()

            metadata = process_file(file_path, write_dir)
            
            elapsed_time = time.time() - start_time
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
            all_metadata.append(metadata)
        except errors.DataError as e:
            print(f'Failed {file_path}: {e}')
    return all_metadata

def process_fn(
        file_path,
        verbose=None,
        write_dir=None):
    try:
        start_time = time.time()

        metadata = process_file(file_path, write_dir)
        
        elapsed_time = time.time() - start_time
        if verbose:
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
        return metadata
    except errors.DataError as e:
        if verbose:
            print(f'Failed {file_path}: {e}')

def main(pdb_dir, write_dir, num_processes, verbose=None):
    
    all_file_paths = glob(f'{pdb_dir}/*/*.pdb')
    # all_file_paths = glob(f'{pdb_dir}/*.pdb')
    total_num_paths = len(all_file_paths)
    write_dir = write_dir

    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    metadata_path = os.path.join(write_dir, 'metadata.csv')
    print(f'\nFiles will be written to {write_dir}.\n')

    # process each pdb file
    if num_processes == 1:
        all_metadata = process_serially(all_file_paths, write_dir)
    else:
        # reduce it down to a single-argument function ...
        _process_fn = fn.partial(process_fn, 
                                 verbose=verbose, 
                                 write_dir=write_dir
        )

        # ... which can then be passed to a parallel pool.map()
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(_process_fn, all_file_paths)

        all_metadata = [x for x in results if x is not None]

    # output metadata csv
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path)
    succeeded = len(metadata_df)
    print(f'Finished processing {succeeded}/{total_num_paths} files.')

if __name__ == "__main__":
    args = parser.parse_args()
    main(
        pdb_dir=args.pdb_dir,
        write_dir=args.write_dir,
        num_processes=args.num_processes,
        verbose=args.verbose
    )
