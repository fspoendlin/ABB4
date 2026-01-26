#!/usr/bin/env python3
"""
Script for renaming chain IDs in PDB files and removing constant regions.
Takes all PDB files in a directory and renames the first chain to H 
and the second chain to L. Also removes all residues with residue number > 128
to keep only variable regions. Useful for preprocessing antibody structures
where chains may have non-standard IDs.

Usage:
python rename_pdb_chains.py --pdb_dir /path/to/pdb/files --output_dir /path/to/output --num_processes 30
"""

import argparse
import os
import glob
import multiprocessing as mp
import functools as fn
from Bio import PDB
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
from tqdm import tqdm


def rename_chains_in_file(file_path, output_dir):
    """
    Rename chains in a single PDB file and remove residues > 128.
    First chain becomes 'H', second chain becomes 'L'.
    Removes all residues with residue number > 128 (keeps variable regions only).
    
    Args:
        file_path: Path to PDB file
        output_dir: Directory to save the renamed file
        
    Returns:
        tuple: (file_path, success, message)
    """
    try:
        # Parse the structure
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure('structure', file_path)
        
        # Get all chains
        chains = list(structure.get_chains())
        
        if len(chains) != 2:
            return (file_path, False, f"Expected 2 chains, found {len(chains)}")
        
        # Check if chains are already H and L
        chain_ids = [chain.id for chain in chains]
        
        # Rename chains using intermediate names to avoid conflicts
        # Step 1: Rename to temporary names
        chains[0].id = '*'  # first chain -> *
        chains[1].id = '-'  # second chain -> -
        
        # Step 2: Rename to final names
        chains[0].id = 'H'  # * -> H
        chains[1].id = 'L'  # - -> L
        
        # Remove residues with residue number > 128 from both chains
        residues_removed = 0
        for chain in chains:
            residues_to_remove = []
            for residue in chain:
                res_id = residue.id[1]  # residue number is the second element of the id tuple
                if res_id > 128:
                    residues_to_remove.append(residue.id)
            
            for res_id in residues_to_remove:
                chain.detach_child(res_id)
                residues_removed += 1
        
        # Write the modified structure
        io = PDB.PDBIO()
        io.set_structure(structure)
        
        # Create output path with same filename in output directory
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)
            
        io.save(output_path)
        
        return (file_path, True, f"Renamed {chain_ids[0]} -> H, {chain_ids[1]} -> L, removed {residues_removed} residues > 128")
        
    except Exception as e:
        return (file_path, False, f"Error: {str(e)}")


def process_file_wrapper(file_path, output_dir, verbose=False):
    """Wrapper function for multiprocessing"""
    result = rename_chains_in_file(file_path, output_dir)
    if verbose:
        print(f"{result[0]}: {result[2]}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Rename chain IDs in PDB files (first -> H, second -> L) and remove residues > 128')
    
    parser.add_argument(
        '--pdb_dir',
        required=True,
        help='Directory containing PDB files',
        type=str)
    
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Directory to save renamed PDB files',
        type=str)
    
    parser.add_argument(
        '--num_processes',
        help='Number of processes for parallel processing',
        type=int,
        default=1)
    
    parser.add_argument(
        '--verbose',
        help='Print progress for each file',
        action='store_true')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    # Find all PDB files
    pdb_pattern = os.path.join(args.pdb_dir, "*.pdb")
    pdb_files = glob.glob(pdb_pattern)
    
    if not pdb_files:
        print(f"No PDB files found in {args.pdb_dir}")
        return
    
    print(f"Found {len(pdb_files)} PDB files in {args.pdb_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Using {args.num_processes} processes")
    
    if args.num_processes == 1:
        # Serial processing
        results = []
        for file_path in tqdm(pdb_files, desc="Processing files"):
            result = process_file_wrapper(file_path, args.output_dir, args.verbose)
            results.append(result)
    else:
        # Parallel processing
        process_fn = fn.partial(
            process_file_wrapper,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        
        with mp.Pool(processes=args.num_processes) as pool:
            if args.verbose:
                results = pool.map(process_fn, pdb_files)
            else:
                results = list(tqdm(
                    pool.imap(process_fn, pdb_files),
                    total=len(pdb_files),
                    desc="Processing files"
                ))
    
    # Summary
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    
    print(f"\nSummary:")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0 and not args.verbose:
        print("\nFailed files:")
        for file_path, success, message in results:
            if not success:
                print(f"  {file_path}: {message}")


if __name__ == "__main__":
    main()
