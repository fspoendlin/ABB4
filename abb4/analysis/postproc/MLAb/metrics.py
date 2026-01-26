import numpy as np

from ABDB.AbPDB import AntibodyParser
from ABDB.ABangle import abangle
from MLAb.util import restype_3to1, chi_angles_atoms, atom_types
from Bio import PDB

# import pdbfixer
# from simtk.openmm import app, LangevinIntegrator, CustomExternalForce
# from simtk import unit

# ENERGY = unit.kilocalories_per_mole
# LENGTH = unit.angstroms
# spring_unit = ENERGY / (LENGTH ** 2)

parser = AntibodyParser()

# https://github.com/biopython/biopython/blob/master/Bio/PDB/DSSP.py
# Wilke: Tien et al. 2013 https://doi.org/10.1371/journal.pone.0080635
SASA_max = {
    "ALA": 129.0,
    "ARG": 274.0,
    "ASN": 195.0,
    "ASP": 193.0,
    "CYS": 167.0,
    "GLN": 225.0,
    "GLU": 223.0,
    "GLY": 104.0,
    "HIS": 224.0,
    "ILE": 197.0,
    "LEU": 201.0,
    "LYS": 236.0,
    "MET": 224.0,
    "PHE": 240.0,
    "PRO": 159.0,
    "SER": 155.0,
    "THR": 172.0,
    "TRP": 285.0,
    "TYR": 263.0,
    "VAL": 174.0,
}

CLASH_CUTOFF = 0.63

# Atomic radii for various atom types.
atom_radii = {"C": 1.70, "N": 1.55, 'O': 1.52, 'S': 1.80}

# Sum of van-der-waals radii
radii_sums = dict(
    [(i + j, (atom_radii[i] + atom_radii[j])) for i in list(atom_radii.keys()) for j in list(atom_radii.keys())])
# Clash_cutoff-based radii values
cutoffs = dict(
    [(i + j, CLASH_CUTOFF * (radii_sums[i + j])) for i in list(atom_radii.keys()) for j in list(atom_radii.keys())])

ab_parser = AntibodyParser(scheme="imgt", definition="imgt")

def calculate_abangles(ab_structure):

    angles = abangle()
    return angles.calculate_angles(ab_structure)


def abangle_errors(true_ab_structure, predicted_ab_structure):
    true_angles = calculate_abangles(true_ab_structure)
    pred_angles = calculate_abangles(predicted_ab_structure)

    return {x: abs(true_angles[x] - pred_angles[x]) for x in true_angles}


def calculate_sasa(ab_structure, probe_radius=1.4, n_points=100, level="R"):
    """ Calculates relative solvent accessible surface area at residue level"""
    # https://github.com/biopython/biopython/blob/master/Bio/PDB/DSSP.py
    # Wilke: Tien et al. 2013 https://doi.org/10.1371/journal.pone.0080635

    sr = PDB.SASA.ShrakeRupley(probe_radius=probe_radius, n_points=n_points)

    fab_structure = ab_structure[0]["HL"]
    fab_structure.level = "M"

    sr.compute(fab_structure, level=level)

    rel_sasas = []
    for res in fab_structure.get_residues():
        rel_sasas.append(res.sasa / SASA_max[res.resname])

    return np.array(rel_sasas)


def exposed_accuracy(true_ab_structure, predicted_ab_structure, exposed_cutoff=0.075):
    true_exposed = calculate_sasa(true_ab_structure) > exposed_cutoff
    pred_exposed = calculate_sasa(predicted_ab_structure) > exposed_cutoff
    # agree = true_exposed == pred_exposed
    agree = np.array([a == b for a, b in zip(true_exposed, pred_exposed)])
    return sum(agree) / len(agree)


def calculate_sidechain_angles(ab_structure):
    angles = np.zeros((len(list(ab_structure.get_residues())), 4))
    angles[...] = float("Nan")
    seq = ""
    for i, res in enumerate(ab_structure.get_residues()):
        if res.id[0] != " ":
            continue
        res_name = restype_3to1[res.resname]
        seq += res_name
        atom_lists = chi_angles_atoms[res_name]
        for j, atom_list in enumerate(atom_lists):
            vec_atoms = [res[a] for a in atom_list]
            vectors = [a.get_vector() for a in vec_atoms]
            angle = PDB.calc_dihedral(*vectors)
            angles[i, j] = angle if angle >= 0 else angle + (2 * np.pi)

    return angles


def sidechain_accuracy(true_ab_structure, predicted_ab_structure):
    true_chis = calculate_sidechain_angles(true_ab_structure)
    pred_chis = calculate_sidechain_angles(predicted_ab_structure)
    diff = np.abs(true_chis - pred_chis)
    diff = np.where(diff >= np.pi, 2 * np.pi - diff, diff)

    forty = np.radians(40)
    chis = {"chi" + str(x + 1): [] for x in range(4)}
    for i in range(len(diff)):
        for j, ang in enumerate(diff[i]):
            if diff[i, j] == diff[i, j]:
                chis["chi" + str(j + 1)].append(diff[i, j] <= forty)

    accuracy = {chi: sum(chis[chi]) / len(chis[chi]) for chi in chis}
    return accuracy


def count_clashes(ab_structure, radius=2.3):

    atoms = [x for x in ab_structure.get_atoms() if x.element in ["C", "N", "O", "S"]]
    coords = np.array([a.coord for a in atoms], dtype="d")
    kdt = PDB.kdtrees.KDTree(coords)

    clashes = []

    for new_atom in atoms:

        kdt_out = kdt.search(np.array(new_atom.coord, dtype="d"), radius)

        inds = [a.index for a in kdt_out]
        dist = [a.radius for a in kdt_out]

        for ix, atom_distance in zip(inds, dist):
            my_atom = atoms[ix]

            # If it's the same residue, ignore
            if new_atom.parent.id == my_atom.parent.id:
                continue

            # We assume C and N don't usually clash unless bonded
            elif (my_atom.name == "C" and new_atom.name == "N") or (my_atom.name == "N" and new_atom.name == "C"):
                continue

            # Ignore disulphide briges
            elif (my_atom.name == "SG" and new_atom.name == "SG") and atom_distance > 1.88:
                continue

            try:
                if atom_distance < (cutoffs[my_atom.element + new_atom.element]):
                    clashes.append((new_atom, my_atom))
            except KeyError:
                continue

    return len(clashes) / 2


def CDR_rmsds(true_ab_structure, predicted_ab_structure):
    rmsds = {}

    for h_or_l in "HL":

        truth = true_ab_structure[0][h_or_l]
        decoy = predicted_ab_structure[0][h_or_l]

        # numb = [x.id for x in decoy.get_residues()]
        numb = [x.id for x in decoy.get_residues() if x in truth.get_residues()]
        
        # Get residues to align
        truth_res = [truth[x] for x in numb if (x in decoy) and (x in truth)]
        decoy_res = [decoy[x] for x in numb if (x in decoy) and (x in truth)]

        # Get atoms to align
        fixed = []
        moved = []
        for i in range(len(decoy_res)):
            fixed += [truth_res[i][atom] for atom in atom_types[:4] if
                      (atom in decoy_res[i]) and (atom in truth_res[i])]
            moved += [decoy_res[i][atom] for atom in atom_types[:4] if
                      (atom in decoy_res[i]) and (atom in truth_res[i])]

        # Calculate superimposer and move decoy
        imposer = PDB.Superimposer()
        imposer.set_atoms(fixed, moved)
        imposer.apply(decoy.get_atoms())
        rmsds[h_or_l] = imposer.rms

        for CDR in "0123":
            true_loop = []
            decoy_loop = []

            for n in numb:

                true_res = truth[n]
                pred_res = decoy[n]

                if CDR != "0":
                    reg = "cdr" + h_or_l.lower() + CDR
                else:
                    reg = "fw" + h_or_l.lower()

                if (reg in true_res.region) and (reg in pred_res.region):
                    assert true_res.resname == pred_res.resname

                    true_loop += [true_res[x].get_coord() for x in atom_types[:4] if
                                  (x in pred_res) and (x in true_res)]
                    decoy_loop += [pred_res[x].get_coord() for x in atom_types[:4] if
                                   (x in pred_res) and (x in true_res)]

            rmsds[reg] = np.sqrt(np.mean(3 * (np.array(true_loop) - np.array(decoy_loop)) ** 2))

    return rmsds

def prediction_metrics(pred_file, true_file, do_sidechain='True'):
    result = {}

    # don't refine for now
    # out_files = {"unrefined": pred_files[j], "refined": refined_files[j]}
    # refine(out_files["unrefined"], out_files["refined"])

    true_ab = ab_parser.get_antibody_structure(true_file, true_file)
    pred_ab = ab_parser.get_antibody_structure(pred_file, pred_file)

    # remove extra residues in truth and pred
    for h_or_l in 'HL':
        true_chain = true_ab[0][h_or_l]
        pred_chain = pred_ab[0][h_or_l]
        extra_resi = [r for r in true_chain.get_residues() if r not in pred_chain.get_residues()]
        if extra_resi != []:
            for resi in extra_resi:
                true_chain.__delitem__(resi.get_id())
        extra_resi = [r for r in pred_chain.get_residues() if r not in true_chain.get_residues()]
        if extra_resi != []:
            for resi in extra_resi:
                pred_chain.__delitem__(resi.get_id())
    assert [r for r in true_ab.get_residues() if r not in pred_ab.get_residues()] == []
    assert [r for r in pred_ab.get_residues() if r not in true_ab.get_residues()] == []

    rmsds = CDR_rmsds(true_ab, pred_ab)
    result.update({x: rmsds[x] for x in rmsds})

    abangles = abangle_errors(true_ab, pred_ab)
    result.update({x: abangles[x] for x in abangles})

    if do_sidechain == 'True':
        side_chain = sidechain_accuracy(true_ab, pred_ab)
        result.update({x: side_chain[x] for x in side_chain})

    result["exposed_accuracy"] = exposed_accuracy(true_ab, pred_ab)
    result["clashes"] = count_clashes(pred_ab)

    return result


# def refine_once(input_file, output_file):
#     # Simple and fast refinement 
#     fixer = pdbfixer.PDBFixer(input_file)

#     fixer.findMissingResidues()
#     fixer.findMissingAtoms()
#     fixer.addMissingAtoms()

#     # Using amber14 recommended protein force field
#     forcefield = app.ForceField("amber14/protein.ff14SB.xml")

#     # Fill in the gaps with OpenMM Modeller
#     modeller = app.Modeller(fixer.topology, fixer.positions)
#     modeller.addHydrogens(forcefield)

#     # Set up force field
#     system = forcefield.createSystem(modeller.topology)

#     # Keep atoms close to initial prediction
#     force = CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
#     force.addGlobalParameter("k", 2.5 * spring_unit)
#     for p in ["x0", "y0", "z0"]:
#         force.addPerParticleParameter(p)

#     for residue in modeller.topology.residues():
#         for atom in residue.atoms():
#             if atom.name in ["CA", "CB", "N", "C"]:
#                 force.addParticle(atom.index, modeller.positions[atom.index])
#     system.addForce(force)

#     # Set up integrator
#     integrator = LangevinIntegrator(100, 0.01, 0.0)

#     # Set up the simulation
#     simulation = app.Simulation(modeller.topology, system, integrator)
#     simulation.context.setPositions(modeller.positions)

#     # Minimize the energy
#     simulation.minimizeEnergy()

#     with open(output_file, "w") as out_handle:
#         app.PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(),
#                               out_handle, keepIds=True)


# def refine(input_file, output_file, n=5):
#     for _ in range(n):
#         try:
#             refine_once(input_file, output_file)
#         except Exception:
#             continue
#         else:
#             break
