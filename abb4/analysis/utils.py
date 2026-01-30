import numpy as np
import os
import re
from abb4.data import protein

custom_colors = ["#020887", "#363CA7", '#88d18a', '#dbd3d8',  "#f7c17a", '#f08700', '#b5446e', "#4E001D", '#FFFFFF', "#4D4D4D"]
custom_palette = {
    'Boltz': custom_colors[0],
    'ABB3': custom_colors[1],
    'AF2 MSA s.': custom_colors[2],
    'aSAM': custom_colors[3],
    'AlphaFlow': custom_colors[4],
    'Bioemu': custom_colors[5],
    'ABB4-STEROIDS': custom_colors[6],
    'ABB4-STEROIDS+': custom_colors[7],
    'GT': custom_colors[8],
    'GT frames': custom_colors[9],
    }

def create_full_prot(
        atom37: np.ndarray,
        atom37_mask: np.ndarray,
        aatype=None,
        b_factors=None,
        imgt_numeric=None,
    ):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
    residue_index = np.arange(n)
    
    # Set chain indices: 0 for Heavy chain (imgt < 1000), 1 for Light chain (imgt >= 1000)
    if imgt_numeric is not None:
        chain_index = np.where(imgt_numeric < 1000, 0, 1)
    else:
        chain_index = np.zeros(n)
    
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors)


def write_prot_to_pdb(
        prot_pos: np.ndarray,
        file_path: str,
        aatype: np.ndarray=None,
        overwrite=False,
        no_indexing=False,
        b_factors=None,
        imgt_numeric=None,
    ):
    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).strip('.pdb')
        existing_files = [x for x in os.listdir(file_dir) if file_name in x]
        max_existing_idx = max([
            int(re.findall(r'_(\d+).pdb', x)[0]) for x in existing_files if re.findall(r'_(\d+).pdb', x)
            if re.findall(r'_(\d+).pdb', x)] + [0])
    if not no_indexing:
        save_path = file_path.replace('.pdb', '') + f'_{max_existing_idx+1}.pdb'
    else:
        save_path = file_path

    if aatype is not None:
        assert aatype.ndim == prot_pos.ndim - 2

    with open(save_path, 'w') as f:
        
        def write_protein_to_file(f, pos37, aatype, b_factors, model, imgt=None):
            atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7
            prot = create_full_prot(pos37, atom37_mask, aatype=aatype.astype(int), b_factors=b_factors, imgt_numeric=imgt)
            pdb_prot = protein.to_pdb(prot, model=model, add_end=False)
            f.write(pdb_prot)

        if prot_pos.ndim == 4:
            for t, pos37 in enumerate(prot_pos):
                write_protein_to_file(f, pos37, aatype[t], b_factors, model=t + 1, imgt=imgt_numeric)
        elif prot_pos.ndim == 3:
            write_protein_to_file(f, prot_pos, aatype, b_factors, model=1, imgt=imgt_numeric)
        else:
            raise ValueError(f'Invalid positions shape {prot_pos.shape}')
        f.write('END')
    return save_path
