"""Convert predicted antibody PDB structures to IMGT numbering using Anarcii.

Iterates over per-target subdirectories under ``--pred_path``, renumbers every
``.pdb`` file found using the Anarcii antibody-numbering model, and writes the
renumbered structures to ``--out_path`` (default: ``<pred_path>/<target>_imgt/``).

Usage
-----
    python scripts/renumber_predictions.py \\
        --pred_path predictions/ \\
        --out_path  predictions_imgt/ \\
        [--gpu]

Arguments
---------
--pred_path : str
    Root directory containing one subdirectory per target, each holding
    ``sample_*.pdb`` files output by inference.
--out_path : str, optional
    Output root directory. If omitted, renumbered files are written next to the
    source directory with an ``_imgt`` suffix.
--gpu : flag
    Run Anarcii on GPU (recommended for large batches). Defaults to CPU.
"""

# from Bio.PDB import PDBParser
from glob import glob
from tqdm import tqdm
import os
from anarcii import Anarcii
# import warnings
import argparse

arg_parser = argparse.ArgumentParser(description="Renumber predicted antibody structures to IMGT numbering using Anarcii")
arg_parser.add_argument('--pred_path', type=str, required=True, help='Path to the predicted PDB file to be renumbered')
arg_parser.add_argument('--out_path', type=str, default=None, help='Output dir for the renumbered PDB files. Default will save to new dir _imgt appended')
arg_parser.add_argument('--gpu', action='store_true', help='Use GPU for Anarcii if available')
args = arg_parser.parse_args()

# long_2_short = {
#     "ALA": "A",
#     "ARG": "R",
#     "ASN": "N",
#     "ASP": "D",
#     "CYS": "C",
#     "GLU": "E",
#     "GLN": "Q",
#     "GLY": "G",
#     "HIS": "H",
#     "ILE": "I",
#     "LEU": "L",
#     "LYS": "K",
#     "MET": "M",
#     "PHE": "F",
#     "PRO": "P",
#     "SER": "S",
#     "THR": "T",
#     "TRP": "W",
#     "TYR": "Y",
#     "VAL": "V"
# }

# pdb_parser = PDBParser()

model = Anarcii(
    seq_type="antibody",
    batch_size=128,
    cpu=not args.gpu,
    ncpu=12,
    mode="accuracy",
    verbose=False, # By setting to True - when we renumber a PDB we can see the renumbered chains.
)

# def get_seperator_idx(seq):
#     seqs = [("seq1", seq)]
#     result = model.number(seqs)['seq1']
#     if result['query_start'] > 20: # second chain recognized
#         sep = result['query_start']
#     else: # first chain recognized
#         sep = result['query_end'] + 1
#     return sep

# def chain_sequence_from_pdb(file_path, chain):
#     '''Extracts the sequence of a chain from a pdb file'''
#     structure = pdb_parser.get_structure('mAb', file_path)
#     residues = structure.get_residues()

#     seq = ''
#     for residue in residues:
#         info = residue.get_full_id()
#         if info[2] == chain:
#             try:
#                 seq += long_2_short[residue.get_resname()]
#             except KeyError:
#                 seq += 'X'
#     return seq

# def rename_2nd_chain(pdb_lines, original_chain, new_chain_label, second_chain_idx=128):
#     pdb_new = []
#     for n, line in enumerate(pdb_lines):
#         if line.startswith('ATOM') or line.startswith('TER'):
#             if line[21] == original_chain and int(line[22:26]) >= second_chain_idx: # where the second domain starts
#                 pdb_new.append(line[:21] + new_chain_label + line[22:])
#             else:
#                 pdb_new.append(line)
#         else:
#             pdb_new.append(line)

#     return pdb_new

# def apply_to_pdb_file(in_path, out_path, funct, **kwargs):
#     '''Applies a function to a pdb file and writes the output to a new file'''
#     with open(in_path, 'r') as f:
#         pdb_lines = f.readlines()

#     pdb_lines = funct(pdb_lines, **kwargs)

#     with open(out_path, 'w') as f:
#         f.write(''.join(pdb_lines))


def main(pred_path, out_path, gpu=False):
    if gpu:
        print("Using GPU for Anarcii.")
    else:
        print("Using CPU for Anarcii.")

    if out_path:
        if not os.path.exists(out_path):
            os.makedirs(out_path)

    dirs = [d for d in glob(f"{pred_path}/*") if os.path.isdir(d)]
    print(f"Found {len(dirs)} directories to process.")
    
    for dir in dirs:

        if out_path:
            pdb_out_stem = out_path + "/" + os.path.basename(dir) + "/"
        else:
            pdb_out_stem = dir + "_imgt/"

        pdbs = glob(f"{dir}/*.pdb")

        count = 0
        for pdb in tqdm(pdbs, desc=f"Renumbering PDBs in {dir}", total=len(pdbs)):
            try:
                os.makedirs(os.path.dirname(pdb_out_stem), exist_ok=True)
                outpath = pdb_out_stem+os.path.basename(pdb)
                _ = model.number(pdb, pdb_out_stem=outpath[0:-4])

                # seq = chain_sequence_from_pdb(pdb, chain='H')
                # sep = get_seperator_idx(seq)
                # apply_to_pdb_file(pdb, outpath, rename_2nd_chain, original_chain='H', new_chain_label='L', second_chain_idx=sep)
                # _ = model.number(outpath, pdb_out_stem=outpath[0:-4])
            
                count += 1
            except Exception as e:
                print(f"Error processing {pdb}: {e}")
                print("Skipping this file.")
        
        print(f"Renumbered {count}/{len(pdbs)} PDB files in {dir} and saved to {pdb_out_stem}")

if __name__ == "__main__":
    main(args.pred_path, out_path=args.out_path, gpu=args.gpu)
