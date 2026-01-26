import torch
import math
from einops import rearrange
import numpy as np

torch.set_default_tensor_type(torch.DoubleTensor)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Residue names definition:
restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}

restype_3to1 = {v: k for k, v in restype_1to3.items()}

# How atoms are sorted in MLAb:

residue_atoms = {
    'A': ['CA', 'N', 'C', 'CB', 'O'],
    'C': ['CA', 'N', 'C', 'CB', 'O', 'SG'],
    'D': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'OD1', 'OD2'],
    'E': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'CD', 'OE1', 'OE2'],
    'F': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'G': ['CA', 'N', 'C', 'CA', 'O'],  # G has no CB so I am padding it with CA so the Os are aligned
    'H': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'CD2', 'CE1', 'ND1', 'NE2'],
    'I': ['CA', 'N', 'C', 'CB', 'O', 'CG1', 'CG2', 'CD1'],
    'K': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'CD', 'CE', 'NZ'],
    'L': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'CD1', 'CD2'],
    'M': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'CE', 'SD'],
    'N': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'ND2', 'OD1'],
    'P': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'CD'],
    'Q': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'CD', 'NE2', 'OE1'],
    'R': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'CD', 'CZ', 'NE', 'NH1', 'NH2'],
    'S': ['CA', 'N', 'C', 'CB', 'O', 'OG'],
    'T': ['CA', 'N', 'C', 'CB', 'O', 'CG2', 'OG1'],
    'V': ['CA', 'N', 'C', 'CB', 'O', 'CG1', 'CG2'],
    'W': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'NE1'],
    'Y': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH']}

residue_atoms_mask = {res: len(residue_atoms[res]) * [True] + (14 - len(residue_atoms[res])) * [False] for res in
                      residue_atoms}

atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]

# Position of atoms in each ref frame

rigid_group_atom_positions2 = {'A': {'C': [0, (1.526, -0.0, -0.0)],
                                     'CA': [0, (0.0, 0.0, 0.0)],
                                     'CB': [0, (-0.529, -0.774, -1.205)],
                                     'N': [0, (-0.525, 1.363, 0.0)],
                                     'O': [3, (-0.627, 1.062, 0.0)]},
                               'C': {'C': [0, (1.524, 0.0, 0.0)],
                                     'CA': [0, (0.0, 0.0, 0.0)],
                                     'CB': [0, (-0.519, -0.773, -1.212)],
                                     'N': [0, (-0.522, 1.362, -0.0)],
                                     'O': [3, (-0.625, 1.062, -0.0)],
                                     'SG': [4, (-0.728, 1.653, 0.0)]},
                               'D': {'C': [0, (1.527, 0.0, -0.0)],
                                     'CA': [0, (0.0, 0.0, 0.0)],
                                     'CB': [0, (-0.526, -0.778, -1.208)],
                                     'CG': [4, (-0.593, 1.398, -0.0)],
                                     'N': [0, (-0.525, 1.362, -0.0)],
                                     'O': [3, (-0.626, 1.062, -0.0)],
                                     'OD1': [5, (-0.61, 1.091, 0.0)],
                                     'OD2': [5, (-0.592, -1.101, 0.003)]},
                               'E': {'C': [0, (1.526, -0.0, -0.0)],
                                     'CA': [0, (0.0, 0.0, 0.0)],
                                     'CB': [0, (-0.526, -0.781, -1.207)],
                                     'CD': [5, (-0.6, 1.397, 0.0)],
                                     'CG': [4, (-0.615, 1.392, 0.0)],
                                     'N': [0, (-0.528, 1.361, 0.0)],
                                     'O': [3, (-0.626, 1.062, 0.0)],
                                     'OE1': [6, (-0.607, 1.095, -0.0)],
                                     'OE2': [6, (-0.589, -1.104, 0.001)]},
                               'F': {'C': [0, (1.524, 0.0, -0.0)],
                                     'CA': [0, (0.0, 0.0, 0.0)],
                                     'CB': [0, (-0.525, -0.776, -1.212)],
                                     'CD1': [5, (-0.709, 1.195, -0.0)],
                                     'CD2': [5, (-0.706, -1.196, 0.0)],
                                     'CE1': [5, (-2.102, 1.198, -0.0)],
                                     'CE2': [5, (-2.098, -1.201, -0.0)],
                                     'CG': [4, (-0.607, 1.377, 0.0)],
                                     'CZ': [5, (-2.794, -0.003, 0.001)],
                                     'N': [0, (-0.518, 1.363, 0.0)],
                                     'O': [3, (-0.626, 1.062, -0.0)]},
                               'G': {'C': [0, (1.517, -0.0, -0.0)],
                                     'CA': [0, (0.0, 0.0, 0.0)],
                                     'N': [0, (-0.572, 1.337, 0.0)],
                                     'O': [3, (-0.626, 1.062, -0.0)]},
                               'H': {'C': [0, (1.525, 0.0, 0.0)],
                                     'CA': [0, (0.0, 0.0, 0.0)],
                                     'CB': [0, (-0.525, -0.778, -1.208)],
                                     'CD2': [5, (-0.889, -1.021, -0.003)],
                                     'CE1': [5, (-2.03, 0.851, -0.002)],
                                     'CG': [4, (-0.6, 1.37, -0.0)],
                                     'N': [0, (-0.527, 1.36, 0.0)],
                                     'ND1': [5, (-0.744, 1.16, -0.0)],
                                     'NE2': [5, (-2.145, -0.466, -0.004)],
                                     'O': [3, (-0.625, 1.063, 0.0)]},
                               'I': {'C': [0, (1.527, -0.0, -0.0)],
                                     'CA': [0, (0.0, 0.0, 0.0)],
                                     'CB': [0, (-0.536, -0.793, -1.213)],
                                     'CD1': [5, (-0.619, 1.391, 0.0)],
                                     'CG1': [4, (-0.534, 1.437, -0.0)],
                                     'CG2': [4, (-0.54, -0.785, 1.199)],
                                     'N': [0, (-0.493, 1.373, -0.0)],
                                     'O': [3, (-0.627, 1.062, -0.0)]},
                               'K': {'C': [0, (1.526, 0.0, 0.0)],
                                     'CA': [0, (0.0, 0.0, 0.0)],
                                     'CB': [0, (-0.524, -0.778, -1.208)],
                                     'CD': [5, (-0.559, 1.417, 0.0)],
                                     'CE': [6, (-0.56, 1.416, 0.0)],
                                     'CG': [4, (-0.619, 1.39, 0.0)],
                                     'N': [0, (-0.526, 1.362, -0.0)],
                                     'NZ': [7, (-0.554, 1.387, 0.0)],
                                     'O': [3, (-0.626, 1.062, -0.0)]},
                               'L': {'C': [0, (1.525, -0.0, -0.0)],
                                     'CA': [0, (0.0, 0.0, 0.0)],
                                     'CB': [0, (-0.522, -0.773, -1.214)],
                                     'CD1': [5, (-0.53, 1.43, -0.0)],
                                     'CD2': [5, (-0.535, -0.774, -1.2)],
                                     'CG': [4, (-0.678, 1.371, 0.0)],
                                     'N': [0, (-0.52, 1.363, 0.0)],
                                     'O': [3, (-0.625, 1.063, -0.0)]},
                               'M': {'C': [0, (1.525, 0.0, 0.0)],
                                     'CA': [0, (0.0, 0.0, 0.0)],
                                     'CB': [0, (-0.523, -0.776, -1.21)],
                                     'CE': [6, (-0.32, 1.786, -0.0)],
                                     'CG': [4, (-0.613, 1.391, -0.0)],
                                     'N': [0, (-0.521, 1.364, -0.0)],
                                     'O': [3, (-0.625, 1.062, -0.0)],
                                     'SD': [5, (-0.703, 1.695, 0.0)]},
                               'N': {'C': [0, (1.526, -0.0, -0.0)],
                                     'CA': [0, (0.0, 0.0, 0.0)],
                                     'CB': [0, (-0.531, -0.787, -1.2)],
                                     'CG': [4, (-0.584, 1.399, 0.0)],
                                     'N': [0, (-0.536, 1.357, 0.0)],
                                     'ND2': [5, (-0.593, -1.188, -0.001)],
                                     'O': [3, (-0.625, 1.062, 0.0)],
                                     'OD1': [5, (-0.633, 1.059, 0.0)]},
                               'P': {'C': [0, (1.527, -0.0, 0.0)],
                                     'CA': [0, (0.0, 0.0, 0.0)],
                                     'CB': [0, (-0.546, -0.611, -1.293)],
                                     'CD': [5, (-0.477, 1.424, 0.0)],
                                     'CG': [4, (-0.382, 1.445, 0.0)],
                                     'N': [0, (-0.566, 1.351, -0.0)],
                                     'O': [3, (-0.621, 1.066, 0.0)]},
                               'Q': {'C': [0, (1.526, 0.0, 0.0)],
                                     'CA': [0, (0.0, 0.0, 0.0)],
                                     'CB': [0, (-0.525, -0.779, -1.207)],
                                     'CD': [5, (-0.587, 1.399, -0.0)],
                                     'CG': [4, (-0.615, 1.393, 0.0)],
                                     'N': [0, (-0.526, 1.361, -0.0)],
                                     'NE2': [6, (-0.593, -1.189, 0.001)],
                                     'O': [3, (-0.626, 1.062, -0.0)],
                                     'OE1': [6, (-0.634, 1.06, 0.0)]},
                               'R': {'C': [0, (1.525, -0.0, -0.0)],
                                     'CA': [0, (0.0, 0.0, 0.0)],
                                     'CB': [0, (-0.524, -0.778, -1.209)],
                                     'CD': [5, (-0.564, 1.414, 0.0)],
                                     'CG': [4, (-0.616, 1.39, -0.0)],
                                     'CZ': [7, (-0.758, 1.093, -0.0)],
                                     'N': [0, (-0.524, 1.362, -0.0)],
                                     'NE': [6, (-0.539, 1.357, -0.0)],
                                     'NH1': [7, (-0.206, 2.301, 0.0)],
                                     'NH2': [7, (-2.078, 0.978, -0.0)],
                                     'O': [3, (-0.626, 1.062, 0.0)]},
                               'S': {'C': [0, (1.525, -0.0, -0.0)],
                                     'CA': [0, (0.0, 0.0, 0.0)],
                                     'CB': [0, (-0.518, -0.777, -1.211)],
                                     'N': [0, (-0.529, 1.36, -0.0)],
                                     'O': [3, (-0.626, 1.062, -0.0)],
                                     'OG': [4, (-0.503, 1.325, 0.0)]},
                               'T': {'C': [0, (1.526, 0.0, -0.0)],
                                     'CA': [0, (0.0, 0.0, 0.0)],
                                     'CB': [0, (-0.516, -0.793, -1.215)],
                                     'CG2': [4, (-0.55, -0.718, 1.228)],
                                     'N': [0, (-0.517, 1.364, 0.0)],
                                     'O': [3, (-0.626, 1.062, 0.0)],
                                     'OG1': [4, (-0.472, 1.353, 0.0)]},
                               'V': {'C': [0, (1.527, -0.0, -0.0)],
                                     'CA': [0, (0.0, 0.0, 0.0)],
                                     'CB': [0, (-0.533, -0.795, -1.213)],
                                     'CG1': [4, (-0.54, 1.429, -0.0)],
                                     'CG2': [4, (-0.533, -0.776, -1.203)],
                                     'N': [0, (-0.494, 1.373, -0.0)],
                                     'O': [3, (-0.627, 1.062, -0.0)]},
                               'W': {'C': [0, (1.525, -0.0, 0.0)],
                                     'CA': [0, (0.0, 0.0, 0.0)],
                                     'CB': [0, (-0.523, -0.776, -1.212)],
                                     'CD1': [5, (-0.824, 1.091, 0.0)],
                                     'CD2': [5, (-0.854, -1.148, -0.005)],
                                     'CE2': [5, (-2.186, -0.678, -0.007)],
                                     'CE3': [5, (-0.622, -2.53, -0.007)],
                                     'CG': [4, (-0.609, 1.37, -0.0)],
                                     'CH2': [5, (-3.028, -2.89, -0.013)],
                                     'CZ2': [5, (-3.283, -1.543, -0.011)],
                                     'CZ3': [5, (-1.715, -3.389, -0.011)],
                                     'N': [0, (-0.521, 1.363, 0.0)],
                                     'NE1': [5, (-2.14, 0.69, -0.004)],
                                     'O': [3, (-0.627, 1.062, 0.0)]},
                               'Y': {'C': [0, (1.524, -0.0, -0.0)],
                                     'CA': [0, (0.0, 0.0, 0.0)],
                                     'CB': [0, (-0.522, -0.776, -1.213)],
                                     'CD1': [5, (-0.716, 1.195, -0.0)],
                                     'CD2': [5, (-0.713, -1.194, -0.001)],
                                     'CE1': [5, (-2.107, 1.2, -0.002)],
                                     'CE2': [5, (-2.104, -1.201, -0.003)],
                                     'CG': [4, (-0.607, 1.382, -0.0)],
                                     'CZ': [5, (-2.791, -0.001, -0.003)],
                                     'N': [0, (-0.522, 1.362, 0.0)],
                                     'O': [3, (-0.627, 1.062, -0.0)],
                                     'OH': [5, (-4.168, -0.002, -0.005)]}}

chi_angles_atoms = {'A': [],
                    'C': [['N', 'CA', 'CB', 'SG']],
                    'D': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
                    'E': [['N', 'CA', 'CB', 'CG'],
                          ['CA', 'CB', 'CG', 'CD'],
                          ['CB', 'CG', 'CD', 'OE1']],
                    'F': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
                    'G': [],
                    'H': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
                    'I': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],
                    'K': [['N', 'CA', 'CB', 'CG'],
                          ['CA', 'CB', 'CG', 'CD'],
                          ['CB', 'CG', 'CD', 'CE'],
                          ['CG', 'CD', 'CE', 'NZ']],
                    'L': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
                    'M': [['N', 'CA', 'CB', 'CG'],
                          ['CA', 'CB', 'CG', 'SD'],
                          ['CB', 'CG', 'SD', 'CE']],
                    'N': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
                    'P': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']],
                    'Q': [['N', 'CA', 'CB', 'CG'],
                          ['CA', 'CB', 'CG', 'CD'],
                          ['CB', 'CG', 'CD', 'OE1']],
                    'R': [['N', 'CA', 'CB', 'CG'],
                          ['CA', 'CB', 'CG', 'CD'],
                          ['CB', 'CG', 'CD', 'NE'],
                          ['CG', 'CD', 'NE', 'CZ']],
                    'S': [['N', 'CA', 'CB', 'OG']],
                    'T': [['N', 'CA', 'CB', 'OG1']],
                    'V': [['N', 'CA', 'CB', 'CG1']],
                    'W': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
                    'Y': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']]}

chi_angles_positions = {}
for r in residue_atoms:
    chi_angles_positions[r] = []
    for angs in chi_angles_atoms[r]:
        chi_angles_positions[r].append([residue_atoms[r].index(atom) for atom in angs])

chi2_centers = {x: chi_angles_atoms[x][1][-2] if len(chi_angles_atoms[x]) > 1 else "CA" for x in chi_angles_atoms}
chi3_centers = {x: chi_angles_atoms[x][2][-2] if len(chi_angles_atoms[x]) > 2 else "CA" for x in chi_angles_atoms}
chi4_centers = {x: chi_angles_atoms[x][3][-2] if len(chi_angles_atoms[x]) > 3 else "CA" for x in chi_angles_atoms}

# This may well be useful in general
rel_pos = {
    x: [rigid_group_atom_positions2[x][residue_atoms[x][atom_id]] if len(residue_atoms[x]) > atom_id else [0, (0, 0, 0)]
        for atom_id in range(14)] for x in rigid_group_atom_positions2}

van_der_waals_radius = {
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "S": 1.8,
}

residue_van_der_waals_radius = {
    x: [van_der_waals_radius[atom[0]] for atom in residue_atoms[x]] + (14 - len(residue_atoms[x])) * ([0]) for x in
    residue_atoms}

valid_rigids = {x: len(chi_angles_atoms[x]) + 2 for x in chi_angles_atoms}


class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.shape = x.shape
        assert (x.shape == y.shape) and (y.shape == z.shape), "x y and z should have the same shape"

    def __add__(self, vec):
        return Vector(vec.x + self.x, vec.y + self.y, vec.z + self.z)

    def __sub__(self, vec):
        return Vector(-vec.x + self.x, -vec.y + self.y, -vec.z + self.z)

    def __mul__(self, param):
        return Vector(param * self.x, param * self.y, param * self.z)

    def __matmul__(self, vec):
        return vec.x * self.x + vec.y * self.y + vec.z * self.z

    def norm(self):
        return (self.x ** 2 + self.y ** 2 + self.z ** 2 + 1e-8) ** (1 / 2)

    def cross(self, other):
        a = (self.y * other.z - self.z * other.y)
        b = (self.z * other.x - self.x * other.z)
        c = (self.x * other.y - self.y * other.x)
        return Vector(a, b, c)

    def dist(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2 + 1e-8) ** (1 / 2)

    def unsqueeze(self, dim):
        return Vector(self.x.unsqueeze(dim), self.y.unsqueeze(dim), self.z.unsqueeze(dim))

    def squeeze(self, dim):
        return Vector(self.x.squeeze(dim), self.y.squeeze(dim), self.z.squeeze(dim))

    def map(self, func):
        return Vector(func(self.x), func(self.y), func(self.z))

    def detach(self):
        return Vector(self.x.detach(), self.y.detach(), self.z.detach())

    def to(self, device):
        return Vector(self.x.to(device), self.y.to(device), self.z.to(device))

    def __str__(self):
        return "Vector(x={},\ny={},\nz={})\n".format(self.x, self.y, self.z)

    def __repr__(self):
        return str(self)

    def __getitem__(self, key):
        return Vector(self.x[key], self.y[key], self.z[key])


class Rot:
    def __init__(self, xx, xy, xz, yx, yy, yz, zx, zy, zz):
        self.xx = xx
        self.xy = xy
        self.xz = xz
        self.yx = yx
        self.yy = yy
        self.yz = yz
        self.zx = zx
        self.zy = zy
        self.zz = zz
        self.shape = xx.shape

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector(
                other.x * self.xx + other.y * self.xy + other.z * self.xz,
                other.x * self.yx + other.y * self.yy + other.z * self.yz,
                other.x * self.zx + other.y * self.zy + other.z * self.zz)

        if isinstance(other, Rot):
            return Rot(
                xx=self.xx * other.xx + self.xy * other.yx + self.xz * other.zx,
                xy=self.xx * other.xy + self.xy * other.yy + self.xz * other.zy,
                xz=self.xx * other.xz + self.xy * other.yz + self.xz * other.zz,
                yx=self.yx * other.xx + self.yy * other.yx + self.yz * other.zx,
                yy=self.yx * other.xy + self.yy * other.yy + self.yz * other.zy,
                yz=self.yx * other.xz + self.yy * other.yz + self.yz * other.zz,
                zx=self.zx * other.xx + self.zy * other.yx + self.zz * other.zx,
                zy=self.zx * other.xy + self.zy * other.yy + self.zz * other.zy,
                zz=self.zx * other.xz + self.zy * other.yz + self.zz * other.zz,
            )

        else:
            raise ValueError("Matmul against {}".format(type(other)))

    def inv(self):
        return Rot(
            xx=self.xx, xy=self.yx, xz=self.zx,
            yx=self.xy, yy=self.yy, yz=self.zy,
            zx=self.xz, zy=self.yz, zz=self.zz
        )

    def det(self):
        return self.xx * self.yy * self.zz + self.xy * self.yz * self.zx + self.yx * self.zy * self.xz - self.xz * self.yy * self.zx - self.xy * self.yx * self.zz - self.xx * self.zy * self.yz

    def unsqueeze(self, dim):
        return Rot(
            self.xx.unsqueeze(dim=dim), self.xy.unsqueeze(dim=dim), self.xz.unsqueeze(dim=dim),
            self.yx.unsqueeze(dim=dim), self.yy.unsqueeze(dim=dim), self.yz.unsqueeze(dim=dim),
            self.zx.unsqueeze(dim=dim), self.zy.unsqueeze(dim=dim), self.zz.unsqueeze(dim=dim)
        )

    def squeeze(self, dim):
        return Rot(
            self.xx.squeeze(dim=dim), self.xy.squeeze(dim=dim), self.xz.squeeze(dim=dim),
            self.yx.squeeze(dim=dim), self.yy.squeeze(dim=dim), self.yz.squeeze(dim=dim),
            self.zx.squeeze(dim=dim), self.zy.squeeze(dim=dim), self.zz.squeeze(dim=dim)
        )

    def detach(self):
        return Rot(
            self.xx.detach(), self.xy.detach(), self.xz.detach(),
            self.yx.detach(), self.yy.detach(), self.yz.detach(),
            self.zx.detach(), self.zy.detach(), self.zz.detach()
        )

    def to(self, device):
        return Rot(
            self.xx.to(device), self.xy.to(device), self.xz.to(device),
            self.yx.to(device), self.yy.to(device), self.yz.to(device),
            self.zx.to(device), self.zy.to(device), self.zz.to(device)
        )

    def __str__(self):
        return "Rot(xx={},\nxy={},\nxz={},\nyx={},\nyy={},\nyz={},\nzx={},\nzy={},\nzz={})\n".format(self.xx, self.xy,
                                                                                                     self.xz, self.yx,
                                                                                                     self.yy, self.yz,
                                                                                                     self.zx, self.zy,
                                                                                                     self.zz)

    def __repr__(self):
        return str(self)

    def __getitem__(self, key):
        return Rot(
            self.xx[key], self.xy[key], self.xz[key],
            self.yx[key], self.yy[key], self.yz[key],
            self.zx[key], self.zy[key], self.zz[key]
        )


class Rigid:
    def __init__(self, origin, rot):
        self.origin = origin
        self.rot = rot
        self.shape = self.origin.shape

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return self.rot @ other + self.origin
        if isinstance(other, Rigid):
            return Rigid(self.rot @ other.origin + self.origin, self.rot @ other.rot)

    def inv(self):
        inv_rot = self.rot.inv()
        t = inv_rot @ self.origin
        return Rigid(Vector(-t.x, -t.y, -t.z), inv_rot)

    def unsqueeze(self, dim=None):
        return Rigid(self.origin.unsqueeze(dim=dim), self.rot.unsqueeze(dim=dim))

    def squeeze(self, dim=None):
        return Rigid(self.origin.squeeze(dim=dim), self.rot.squeeze(dim=dim))

    def detach(self):
        return Rigid(self.origin.detach(), self.rot.detach())

    def to(self, device):
        return Rigid(self.origin.to(device), self.rot.to(device))

    def __str__(self):
        return "Rigid(origin={},\nrot={})".format(self.origin, self.rot)

    def __repr__(self):
        return str(self)

    def __getitem__(self, key):
        return Rigid(self.origin[key], self.rot[key])


def vec_from_tensor(tens):
    assert tens.shape[-1] == 3, "What dimension you in?"
    return Vector(tens[..., 0], tens[..., 1], tens[..., 2])


def rigid_from_three_points(origin, y_x_plane, x_axis):
    v1 = x_axis - origin
    v2 = y_x_plane - origin

    v1 *= 1 / v1.norm()
    v2 = v2 - v1 * (v1 @ v2)
    v2 *= 1 / v2.norm()
    v3 = v1.cross(v2)
    rot = Rot(v1.x, v2.x, v3.x, v1.y, v2.y, v3.y, v1.z, v2.z, v3.z)
    return Rigid(origin, rot)


def rigid_from_tensor(tens):
    assert (tens.shape[-1] == 3), "I want 3D points"
    return rigid_from_three_points(vec_from_tensor(tens[..., 0, :]), vec_from_tensor(tens[..., 1, :]),
                                   vec_from_tensor(tens[..., 2, :]))


def backbone_from_rigid(rig, seq):
    positions = [rigid_group_atom_positions2[x] for x in seq]
    coords = torch.zeros((*rig.origin.x.shape, 3, 3))

    for i, atom in enumerate(["CA", "N", "C"]):
        atoms = rig.to(device) @ vec_from_tensor(torch.tensor([list(x[atom][1]) for x in positions]).to(device))
        coords[..., i, :] = torch.stack([atoms.x, atoms.y, atoms.z], dim=-1)
    return coords


def BB_fape(rig1, rig2):
    o1, o2 = rig1.origin.unsqueeze(-1), rig2.origin.unsqueeze(-1)
    x1, x2 = rig1.inv() @ o1, rig2.inv() @ o2
    return (x1.dist(x2) / 10).clamp(max=1.0).mean()


def generic_fape(rig1, rig2, points1, points2, clamped=True):
    x1, x2 = rig1.inv() @ vec_from_tensor(points1).unsqueeze(-1).unsqueeze(-1), rig2.inv() @ vec_from_tensor(
        points2).unsqueeze(-1).unsqueeze(-1)
    if clamped:
        return (x1.dist(x2) / 10).clamp(max=1.0).mean()
    else:
        return x1.dist(x2).mean() / 10


def rotate_x_axis_to_new_vector(new_vector):
    c, b, a = new_vector[..., 0], new_vector[..., 1], new_vector[..., 2]
    n = (c ** 2 + a ** 2 + b ** 2 + 1e-16) ** (1 / 2)

    a, b, c = -a / n, b / n, c / n
    k = (1 - c) / (a ** 2 + b ** 2 + 1e-8)

    new_origin = Vector(torch.zeros_like(a), torch.zeros_like(a), torch.zeros_like(a))
    new_rot = Rot(c, -b, a, b, 1 - k * b ** 2, a * b * k, -a, a * b * k, 1 - k * a ** 2)

    return Rigid(new_origin, new_rot)


def stack_rigids(rigids, **kwargs):
    # Probably best to avoid using very much
    stacked_origin = Vector(torch.stack([rig.origin.x for rig in rigids], **kwargs),
                            torch.stack([rig.origin.y for rig in rigids], **kwargs),
                            torch.stack([rig.origin.z for rig in rigids], **kwargs))
    stacked_rot = Rot(
        torch.stack([rig.rot.xx for rig in rigids], **kwargs), torch.stack([rig.rot.xy for rig in rigids], **kwargs),
        torch.stack([rig.rot.xz for rig in rigids], **kwargs),
        torch.stack([rig.rot.yx for rig in rigids], **kwargs), torch.stack([rig.rot.yy for rig in rigids], **kwargs),
        torch.stack([rig.rot.yz for rig in rigids], **kwargs),
        torch.stack([rig.rot.zx for rig in rigids], **kwargs), torch.stack([rig.rot.zy for rig in rigids], **kwargs),
        torch.stack([rig.rot.zz for rig in rigids], **kwargs),
    )
    return Rigid(stacked_origin, stacked_rot)


def rot_matrices_from_geom(geom, atom_groups):
    bases = []
    rots = []
    for i in range(len(geom)):
        bases.append(
            rearrange(geom[i, [atom_groups[i, :, 1], atom_groups[i, :, 0], atom_groups[i, :, 2]]], "i j k -> j i k"))
        rots.append(
            rearrange(geom[i, [atom_groups[i, :, 2], atom_groups[i, :, 3], atom_groups[i, :, 1]]], "i j k -> j i k"))
    return rigid_from_tensor(torch.stack(bases, dim=0)).inv() @ rigid_from_tensor(torch.stack(rots, dim=0))


def rigid_body_identity(shape):
    return Rigid(Vector(*3 * [torch.zeros(shape)]),
                 Rot(torch.ones(shape), *3 * [torch.zeros(shape)], torch.ones(shape), *3 * [torch.zeros(shape)],
                     torch.ones(shape)))


def global_frames_from_geom(geom, seq):
    atom_groups = np.array(
        [[[0, 2, 0, 1], [1, 0, 2, 4]] + chi_angles_positions[x] + (4 - len(chi_angles_positions[x])) * [[0, 0, 0, 0]]
         for x in seq])
    global_frames = []
    for i in range(len(geom)):
        global_frames.append(
            rearrange(geom[i, [atom_groups[i, :, 2], atom_groups[i, :, 3], atom_groups[i, :, 1]]], "i j k -> j i k"))
    return rigid_from_tensor(torch.stack(global_frames, dim=0))


def torsion_angles_from_geom(geom, seq):
    atom_groups = np.array(
        [[[1, 0, 2, 4]] + chi_angles_positions[x] + (4 - len(chi_angles_positions[x])) * [[0, 0, 0, 0]] for x in seq])
    rot_mats = rot_matrices_from_geom(geom, atom_groups).rot
    return torch.stack([rot_mats.yy, rot_mats.yz], dim=-1)


def obtain_alignment(fixed, to_be_aligned):
    device = fixed.x.device
    p1 = to_be_aligned - to_be_aligned.map(torch.mean)
    p2 = fixed - fixed.map(torch.mean)

    X_ = torch.stack([p1.x, p1.y, p1.z], dim=-1)
    Y_ = torch.stack([p2.x, p2.y, p2.z], dim=-1)

    C = torch.matmul(X_.t(), Y_)
    V, _, W = torch.linalg.svd(C)
    U = torch.matmul(V, W)

    if torch.det(U) < 0:
        U = torch.matmul(torch.tensor([[1, 1, -1]], device=device) * V, W)

    rot = Rot(*U.t().flatten())
    origin = fixed.map(torch.mean) - (rot @ to_be_aligned).map(torch.mean)

    return Rigid(origin=origin, rot=rot).to(device)


def rigid_transformation_from_torsion_angles(torsion_angles, distance_to_new_origin):
    dev = torsion_angles.device

    zero = torch.zeros(torsion_angles.shape[:-1]).to(dev)
    one = torch.ones(torsion_angles.shape[:-1]).to(dev)
    new_rot = Rot(
        -one, zero, zero,
        zero, torsion_angles[..., 0], torsion_angles[..., 1],
        zero, torsion_angles[..., 1], -torsion_angles[..., 0],
    )
    new_origin = Vector(distance_to_new_origin, zero, zero)

    return Rigid(new_origin, new_rot)


def rotate_x_axis_to_new_vector(new_vector):
    # Extract coordinates
    c, b, a = new_vector[..., 0], new_vector[..., 1], new_vector[..., 2]

    # Normalize
    n = (c ** 2 + a ** 2 + b ** 2 + 1e-16) ** (1 / 2)
    a, b, c = a / n, b / n, -c / n

    # Set new origin
    new_origin = vec_from_tensor(torch.zeros_like(new_vector))

    # Rotate x-axis to point old origin to new one
    k = (1 - c) / (a ** 2 + b ** 2 + 1e-8)
    new_rot = Rot(-c, b, -a, b, 1 - k * b ** 2, a * b * k, a, -a * b * k, k * a ** 2 - 1)

    return Rigid(new_origin, new_rot)


def global_frames_from_bb_frame_and_torsion_angles(bb_frame, torsion_angles, seq):
    dev = bb_frame.origin.x.device

    # We start with psi
    psi_local_frame_origin = torch.tensor([rel_pos[x][2][1] for x in seq]).to(dev).pow(2).sum(-1).pow(1 / 2)
    psi_local_frame = rigid_transformation_from_torsion_angles(torsion_angles[:, 0], psi_local_frame_origin)
    psi_global_frame = bb_frame @ psi_local_frame

    # Now all the chis
    chi1_local_frame_origin = torch.tensor([rel_pos[x][3][1] for x in seq]).to(dev)
    chi1_local_frame = rotate_x_axis_to_new_vector(chi1_local_frame_origin) @ rigid_transformation_from_torsion_angles(
        torsion_angles[:, 1], chi1_local_frame_origin.pow(2).sum(-1).pow(1 / 2))
    chi1_global_frame = bb_frame @ chi1_local_frame

    chi2_local_frame_origin = torch.tensor([rigid_group_atom_positions2[x][chi2_centers[x]][1] for x in seq]).to(dev)
    chi2_local_frame = rotate_x_axis_to_new_vector(chi2_local_frame_origin) @ rigid_transformation_from_torsion_angles(
        torsion_angles[:, 2], chi2_local_frame_origin.pow(2).sum(-1).pow(1 / 2))
    chi2_global_frame = chi1_global_frame @ chi2_local_frame

    chi3_local_frame_origin = torch.tensor([rigid_group_atom_positions2[x][chi3_centers[x]][1] for x in seq]).to(dev)
    chi3_local_frame = rotate_x_axis_to_new_vector(chi3_local_frame_origin) @ rigid_transformation_from_torsion_angles(
        torsion_angles[:, 3], chi3_local_frame_origin.pow(2).sum(-1).pow(1 / 2))
    chi3_global_frame = chi2_global_frame @ chi3_local_frame

    chi4_local_frame_origin = torch.tensor([rigid_group_atom_positions2[x][chi4_centers[x]][1] for x in seq]).to(dev)
    chi4_local_frame = rotate_x_axis_to_new_vector(chi4_local_frame_origin) @ rigid_transformation_from_torsion_angles(
        torsion_angles[:, 4], chi4_local_frame_origin.pow(2).sum(-1).pow(1 / 2))
    chi4_global_frame = chi3_global_frame @ chi4_local_frame

    return stack_rigids(
        [bb_frame, psi_global_frame, chi1_global_frame, chi2_global_frame, chi3_global_frame, chi4_global_frame],
        dim=-1)


def all_atoms_from_global_reference_frames(global_reference_frames, seq):
    dev = global_reference_frames.origin.x.device

    all_atoms = torch.zeros((len(seq), 14, 3)).to(dev)
    for atom_pos in range(14):
        relative_positions = [rel_pos[x][atom_pos][1] for x in seq]
        local_reference_frame = [max(rel_pos[x][atom_pos][0] - 2, 0) for x in seq]
        local_reference_frame_mask = torch.tensor([[y == x for y in range(6)] for x in local_reference_frame]).to(dev)
        global_atom_vector = global_reference_frames[local_reference_frame_mask] @ vec_from_tensor(
            torch.tensor(relative_positions).to(dev))
        all_atoms[:, atom_pos] = torch.stack([global_atom_vector.x, global_atom_vector.y, global_atom_vector.z], dim=-1)

    all_atom_mask = torch.tensor([residue_atoms_mask[x] for x in seq]).to(dev)
    all_atoms[~all_atom_mask] = float("Nan")
    return all_atoms, all_atom_mask


class GatedAttention(torch.nn.Module):
    def __init__(self, node_dim, heads=8, head_dim=32):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim

        attention_inner_dim = heads * head_dim

        self.first_norm = torch.nn.LayerNorm(node_dim)
        self.qkv = torch.nn.Linear(node_dim, 3 * attention_inner_dim, bias=False)

        self.gate = torch.nn.Linear(node_dim, attention_inner_dim)

        self.final = torch.nn.Linear(attention_inner_dim, node_dim)

        with torch.no_grad():
            self.final.weight.fill_(0.0)
            self.final.bias.fill_(0.0)
            self.gate.weight.fill_(0.0)
            self.gate.bias.fill_(1.0)

    def forward(self, node_features):
        normed_features = self.first_norm(node_features)

        qkv = self.qkv(normed_features).chunk(3, dim=-1)
        query, key, value = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        gate = rearrange(self.gate(node_features), 'b n (h d) -> b h n d', h=self.heads).sigmoid()

        attention_matrix = torch.einsum('b h i d, b h j d -> b h i j', query, key) * self.head_dim ** (-1 / 2)

        output = torch.einsum('b h i j, b h i d -> b h j d', attention_matrix.softmax(-1), value) * gate

        return node_features + self.final(rearrange(output, 'b h n d -> b n (h d)'))


class GatedAttentionWithPairBias(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, heads=8, head_dim=32):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim

        attention_inner_dim = heads * head_dim

        self.first_norm = torch.nn.LayerNorm(node_dim)
        self.qkv = torch.nn.Linear(node_dim, 3 * attention_inner_dim, bias=False)

        self.pair_bias = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_dim),
            torch.nn.Linear(edge_dim, heads, bias=False)
        )

        self.gate = torch.nn.Linear(node_dim, attention_inner_dim)

        self.final = torch.nn.Linear(attention_inner_dim, node_dim)

        with torch.no_grad():
            self.final.weight.fill_(0.0)
            self.final.bias.fill_(0.0)
            self.gate.weight.fill_(0.0)
            self.gate.bias.fill_(1.0)

    def forward(self, node_features, edge_features):
        normed_features = self.first_norm(node_features)

        qkv = self.qkv(normed_features).chunk(3, dim=-1)
        query, key, value = map(lambda t: rearrange(t, 'n (h d) -> h n d', h=self.heads), qkv)

        pair_bias = rearrange(self.pair_bias(edge_features), 'i j h -> h i j')
        gate = rearrange(self.gate(node_features), 'n (h d) -> h n d', h=self.heads).sigmoid()

        attention_matrix = torch.einsum('h i d, h j d -> h i j', query, key) * self.head_dim ** (-1 / 2) + pair_bias

        output = torch.einsum('h i j, h i d -> h j d', attention_matrix.softmax(-1), value) * gate

        return node_features + self.final(rearrange(output, 'h n d -> n (h d)'))


class AxialAttention(torch.nn.Module):
    def __init__(self, edge_dim, heads=4, head_dim=16):
        super().__init__()

        self.row_attention = GatedAttention(edge_dim, heads, head_dim)
        self.column_attention = GatedAttention(edge_dim, heads, head_dim)

        self.ff = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(edge_dim, 2 * edge_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * edge_dim, edge_dim)
        )

        with torch.no_grad():
            self.ff.weight.fill_(0.0)

    def forward(self, edge_features):
        edge_features = self.row_attention(edge_features)
        edge_features = rearrange(edge_features, 'i j d -> j i d')
        edge_features = self.column_attention(edge_features)
        edge_features = rearrange(edge_features, 'j i d -> i j d')

        return edge_features + self.ff(edge_features)


class OuterProductSum(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, inner_dim=32):
        super().__init__()

        self.embed = torch.nn.Sequential(
            torch.nn.LayerNorm(node_dim),
            torch.nn.Linear(node_dim, 2 * inner_dim)
        )
        self.final = torch.nn.Linear(inner_dim ** 2, edge_dim)

        with torch.no_grad():
            self.final.weight.fill_(0.0)
            self.final.bias.fill_(0.0)

    def forward(self, node_features, edge_features):
        a, b = self.embed(node_features).chunk(2, dim=-1)
        outter = torch.einsum('i d, j k -> i j d k', a, b).flatten(start_dim=2)

        return edge_features + self.final(outter)


class PairUpdate(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, inner_dim=32, heads=8, head_dim=32):
        super().__init__()

        # self.axial_attention = AxialAttention(edge_dim, heads, head_dim)
        self.outer_product = OuterProductSum(node_dim, edge_dim, inner_dim)

    def forward(self, node_features, edge_features):
        mixed = self.outer_product(node_features, edge_features)

        return mixed  # self.axial_attention(mixed)


class SequenceBlock(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, inner_dim=8, heads=4, head_dim=8, edge_update=False, **kwargs):
        super().__init__()
        self.edge_update = edge_update
        self.pair_update = PairUpdate(node_dim, edge_dim, inner_dim, heads, head_dim)
        self.sequence_update = GatedAttentionWithPairBias(node_dim, edge_dim, heads, head_dim)

    def forward(self, node_features, edge_features):
        node_features = self.sequence_update(node_features, edge_features)
        if self.edge_update:
            edge_features = self.pair_update(node_features, edge_features)

        return node_features, edge_features


class InvariantPointAttention(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, heads=12, head_dim=16, n_query_points=4, n_value_points=8, **kwargs):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.n_query_points = n_query_points

        node_scalar_attention_inner_dim = heads * head_dim
        node_vector_attention_inner_dim = 3 * n_query_points * heads
        node_vector_attention_value_dim = 3 * n_value_points * heads
        after_final_cat_dim = heads * edge_dim + heads * head_dim + heads * n_value_points * 4

        point_weight_init_value = torch.log(torch.exp(torch.full((heads,), 1.)) - 1.)
        self.point_weight = torch.nn.Parameter(point_weight_init_value)

        self.to_scalar_qkv = torch.nn.Linear(node_dim, 3 * node_scalar_attention_inner_dim, bias=False)
        self.to_vector_qk = torch.nn.Linear(node_dim, 2 * node_vector_attention_inner_dim, bias=False)
        self.to_vector_v = torch.nn.Linear(node_dim, node_vector_attention_value_dim, bias=False)
        self.to_scalar_edge_attention_bias = torch.nn.Linear(edge_dim, heads, bias=False)
        self.final_linear = torch.nn.Linear(after_final_cat_dim, node_dim)

        with torch.no_grad():
            self.final_linear.weight.fill_(0.0)
            self.final_linear.bias.fill_(0.0)

    def forward(self, node_features, edge_features, rigid):
        # Classic attention on nodes
        scalar_qkv = self.to_scalar_qkv(node_features).chunk(3, dim=-1)
        scalar_q, scalar_k, scalar_v = map(lambda t: rearrange(t, 'n (h d) -> h n d', h=self.heads), scalar_qkv)
        node_scalar = torch.einsum('h i d, h j d -> h i j', scalar_q, scalar_k) * self.head_dim ** (-1 / 2)

        # Linear bias on edges
        edge_bias = rearrange(self.to_scalar_edge_attention_bias(edge_features), 'i j h -> h i j')

        # Reference frame attention
        wc = (2 / self.n_query_points) ** (1 / 2) / 6
        vector_qk = self.to_vector_qk(node_features).chunk(2, dim=-1)
        vector_q, vector_k = map(lambda x: vec_from_tensor(rearrange(x, 'n (h p d) -> h n p d', h=self.heads, d=3)),
                                 vector_qk)
        rigid_ = rigid.unsqueeze(0).unsqueeze(-1)  # add head and point dimension to rigids

        global_vector_k = rigid_ @ vector_k
        global_vector_q = rigid_ @ vector_q
        global_frame_distance = wc * global_vector_q.unsqueeze(-2).dist(global_vector_k.unsqueeze(-3)).sum(
            -1) * rearrange(self.point_weight, "h -> h () ()")

        # Combining attentions
        attention_matrix = (3 ** (-1 / 2) * (node_scalar + edge_bias - global_frame_distance)).softmax(-1)

        # Obtaining outputs
        edge_output = (rearrange(attention_matrix, 'h i j -> i h () j') * rearrange(edge_features,
                                                                                    'i j d -> i () d j')).sum(-1)
        scalar_node_output = torch.einsum('h i j, h j d -> i h d', attention_matrix, scalar_v)

        vector_v = vec_from_tensor(
            rearrange(self.to_vector_v(node_features), 'n (h p d) -> h n p d', h=self.heads, d=3))
        global_vector_v = rigid_ @ vector_v
        attended_global_vector_v = global_vector_v.map(
            lambda x: torch.einsum('h i j, h j p -> h i p', attention_matrix, x))
        vector_node_output = rigid_.inv() @ attended_global_vector_v
        vector_node_output = torch.stack(
            [vector_node_output.norm(), vector_node_output.x, vector_node_output.y, vector_node_output.z], dim=-1)

        # Concatenate along heads and points
        edge_output = rearrange(edge_output, 'n h d -> n (h d)')
        scalar_node_output = rearrange(scalar_node_output, 'n h d -> n (h d)')
        vector_node_output = rearrange(vector_node_output, 'h n p d -> n (h p d)')

        combined = torch.cat([edge_output, scalar_node_output, vector_node_output], dim=-1)

        return node_features + self.final_linear(combined)


class BackboneUpdate(torch.nn.Module):
    def __init__(self, node_dim):
        super().__init__()

        self.to_correction = torch.nn.Linear(node_dim, 6)

    def forward(self, node_features, update_mask=None):
        # Predict quaternions and translation vector
        rot, t = self.to_correction(node_features).chunk(2, dim=-1)

        # I may not want to update all residues
        if update_mask is not None:
            rot = update_mask[:, None] * rot
            t = update_mask[:, None] * t

        # Normalize quaternions
        norm = (1 + rot.pow(2).sum(-1, keepdim=True)).pow(1 / 2)
        b, c, d = (rot / norm).chunk(3, dim=-1)
        a = 1 / norm
        a, b, c, d = a.squeeze(-1), b.squeeze(-1), c.squeeze(-1), d.squeeze(-1)

        # Make rotation matrix from quaternions
        R = Rot(
            (a ** 2 + b ** 2 - c ** 2 - d ** 2), (2 * b * c - 2 * a * d), (2 * b * d + 2 * a * c),
            (2 * b * c + 2 * a * d), (a ** 2 - b ** 2 + c ** 2 - d ** 2), (2 * c * d - 2 * a * b),
            (2 * b * d - 2 * a * c), (2 * c * d + 2 * a * b), (a ** 2 - b ** 2 - c ** 2 + d ** 2)
        )

        return Rigid(vec_from_tensor(t), R)


class TorsionAngles(torch.nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.residual1 = torch.nn.Sequential(
            torch.nn.Linear(2 * node_dim, 2 * node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * node_dim, 2 * node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * node_dim, 2 * node_dim)
        )

        self.residual2 = torch.nn.Sequential(
            torch.nn.Linear(2 * node_dim, 2 * node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * node_dim, 2 * node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * node_dim, 2 * node_dim)
        )

        self.final_pred = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(2 * node_dim, 10)
        )

        with torch.no_grad():
            self.residual1[-1].weight.fill_(0.0)
            self.residual2[-1].weight.fill_(0.0)
            self.residual1[-1].bias.fill_(0.0)
            self.residual2[-1].bias.fill_(0.0)

    def forward(self, node_features, s_i):
        full_feat = torch.cat([node_features, s_i], axis=-1)

        full_feat = full_feat + self.residual1(full_feat)
        full_feat = full_feat + self.residual2(full_feat)
        torsions = rearrange(self.final_pred(full_feat), "i (t d) -> i t d", d=2)
        norm = torch.norm(torsions, dim=-1, keepdim=True)

        return torsions / norm, norm


class StructureUpdate(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, propagate_rotation_gradient=False, dropout=0.1, **kwargs):
        super().__init__()
        self.propagate_rotation_gradient = propagate_rotation_gradient

        self.IPA = InvariantPointAttention(node_dim, edge_dim, **kwargs)
        self.norm1 = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.LayerNorm(node_dim)
        )
        self.norm2 = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.LayerNorm(node_dim)
        )
        self.residual = torch.nn.Sequential(
            torch.nn.Linear(node_dim, 2 * node_dim),  # Pulling these dims out of nowhere
            torch.nn.ReLU(),
            torch.nn.Linear(2 * node_dim, 2 * node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * node_dim, node_dim)
        )

        self.torsion_angles = TorsionAngles(node_dim)
        self.backbone_update = BackboneUpdate(node_dim)

        with torch.no_grad():
            self.residual[-1].weight.fill_(0.0)
            self.residual[-1].bias.fill_(0.0)

    def forward(self, node_features, edge_features, rigid_pred, update_mask=None):
        s_i = self.IPA(node_features, edge_features, rigid_pred)
        s_i = self.norm1(s_i)
        s_i = s_i + self.residual(s_i)
        s_i = self.norm2(s_i)
        rigid_new = rigid_pred @ self.backbone_update(s_i, update_mask)

        if not self.propagate_rotation_gradient:
            rigid_new = Rigid(rigid_new.origin, rigid_new.rot.detach())

        return s_i, rigid_new


class ErrorEstimate(torch.nn.Module):
    def __init__(self, node_dim=23, edge_features=16, layers=2, **kwargs):
        super().__init__()
        self.ipa_layers = torch.nn.ModuleList(
            [InvariantPointAttention(node_dim, edge_features, **kwargs) for _ in range(layers)])
        self.reduction_MLP = torch.nn.Sequential(
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim, node_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim // 2, 1)
        )

    def forward(self, node_features, edge_features, rigid):
        new_node_features = node_features
        for layer in self.ipa_layers:
            new_node_features = layer(new_node_features, edge_features, rigid)

        return self.reduction_MLP(new_node_features).squeeze(-1)


class StructureModule(torch.nn.Module):
    def __init__(self, node_dim, n_layers=4, rel_pos_dim=16, embed_dim=32, error_estimation=False, n_seq_blocks=0, extra_losses=True, rotation_prop=False,
                 **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.rel_pos_dim = rel_pos_dim
        self.extra_losses = extra_losses

        self.node_embed = torch.nn.Linear(node_dim, embed_dim)
        self.edge_embed = torch.nn.Linear(2 * rel_pos_dim + 1, embed_dim - 1)

        if error_estimation:
            self.error_estimation = ErrorEstimate(embed_dim, embed_dim)
        else:
            self.error_estimation = False

        if n_seq_blocks > 0:
            self.seq_blocks = torch.nn.ModuleList(
                [SequenceBlock(embed_dim, embed_dim - 1, **kwargs) for _ in range(n_seq_blocks)])
        else:
            self.seq_blocks = []

        self.layers = torch.nn.ModuleList(
            [StructureUpdate(node_dim=embed_dim, edge_dim=embed_dim, propagate_rotation_gradient=(i == n_layers - 1) or rotation_prop,
                             **kwargs) for i in range(n_layers)])

        self.aux_loss = 0

    def forward(self, node_features, rigid_in, sequence, rigid_true=None, torsion_true=None, update_mask=None,
                CDR_mask=None):
        self.aux_loss = 0
        aux_fape_loss = 0
        aux_torsion_loss = 0
        norm_loss = 0
        relative_positions = (torch.arange(node_features.shape[-2])[None] -
                              torch.arange(node_features.shape[-2])[:, None]).clamp(min=-self.rel_pos_dim,
                                                                                    max=self.rel_pos_dim) + self.rel_pos_dim
        rel_pos_embedings = self.edge_embed(
            torch.nn.functional.one_hot(relative_positions, num_classes=2 * self.rel_pos_dim + 1).to(
                dtype=node_features.dtype, device=node_features.device))

        new_node_features = self.node_embed(node_features)

        for seq_block in self.seq_blocks:
            new_node_features, rel_pos_embedings = seq_block(new_node_features, rel_pos_embedings)

        torsion_mask = torsion_true.norm(dim=-1, keepdim=True)
        for layer in self.layers:
            edge_features = torch.cat(
                [rigid_in.origin.unsqueeze(-1).dist(rigid_in.origin).unsqueeze(-1), rel_pos_embedings], dim=-1)
            new_node_features, rigid_in = layer(new_node_features, edge_features, rigid_in, update_mask)
            torsions, norm = layer.torsion_angles(self.node_embed(node_features), new_node_features)
            norm_loss = 0.02 * (1 - norm).abs().mean()

            if self.extra_losses:
                if rigid_true is not None:
                    aux_fape_loss = aux_fape_loss + BB_fape(rigid_in, rigid_true) / (2 * self.n_layers)
                if torsion_true is not None:
                    aux_torsion_loss = aux_torsion_loss + ((torsions - torsion_true) * torsion_mask).pow(2).sum(
                        -1).mean() / (2 * self.n_layers)

        self.aux_loss = (aux_fape_loss + aux_torsion_loss + norm_loss)
        self.aux_loss = self.aux_loss + (
                BB_fape(rigid_in, rigid_true) + ((torsions - torsion_true) * torsion_mask).pow(2).sum(-1).mean()) / 2

        if self.error_estimation and (CDR_mask is not None):
            error = rigid_true.origin.dist(
                obtain_alignment(rigid_true.origin[~CDR_mask], rigid_in.origin[~CDR_mask]) @ rigid_in.origin).detach()
            predicted_error = self.error_estimation(new_node_features, edge_features.detach(), rigid_in.detach())
            self.aux_loss = self.aux_loss + 0.01 * ((predicted_error - error).pow(2)).mean().sqrt()

        all_reference_frames = global_frames_from_bb_frame_and_torsion_angles(rigid_in, torsions, sequence)
        all_atoms, all_atom_mask = all_atoms_from_global_reference_frames(all_reference_frames, sequence)

        return new_node_features, all_reference_frames, all_atoms, all_atom_mask


class RAdam(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


def rigid_in_from_geom_out(geom, encodings):
    mask = np.equal(geom[:, :3], geom[:, :3]).all(-1).all(-1)
    return rigid_body_identity(geom.shape[:-2]), torch.tensor(mask)


def rmsd(prediction, truth):
    dists = (prediction - truth).pow(2).sum(-1)
    return torch.sqrt(dists.mean())


def dist_between_ca_ca(points):
    return (points[1:, 0] - points[:-1, 0]).pow(2).sum(-1).pow(1 / 2).clamp(max=10)


def dist_between_c_n(points):
    return (points[1:, 1] - points[:-1, 2]).pow(2).sum(-1).pow(1 / 2).clamp(max=10)


def ang_between_ca_n_c(points):
    can = points[1:, 0] - points[1:, 1]
    cn = points[1:, 1] - points[:-1, 2]
    return (can * cn).sum(-1) / (can.norm(dim=-1) * cn.norm(dim=-1))


def ang_between_n_c_ca(points):
    cac = points[:-1, 0] - points[:-1, 2]
    cn = points[1:, 1] - points[:-1, 2]
    return (cac * cn).sum(-1) / (cac.norm(dim=-1) * cn.norm(dim=-1))


def dist_check(pred_points, true_points, k=12):
    ca_ca_dist = ((dist_between_ca_ca(pred_points) - dist_between_ca_ca(true_points)).abs() - 0.075 * k).clamp(0).mean()
    c_n_dist = ((dist_between_c_n(pred_points) - dist_between_c_n(true_points)).abs() - 0.04 * k).clamp(0).mean()
    return (ca_ca_dist + c_n_dist) / 4


def angle_check(pred_points, true_points, k=12):
    ca_n_c = ((ang_between_ca_n_c(pred_points) - ang_between_ca_n_c(true_points)).abs() - 0.02 * k).clamp(0).mean()
    n_c_ca = ((ang_between_n_c_ca(pred_points) - ang_between_n_c_ca(true_points)).abs() - 0.02 * k).clamp(0).mean()
    return (n_c_ca + ca_n_c) / 4


def get_torsions(ab, cb, db):
    """Vectors go backwards so ab is p2 - p1, cd is p3 - p2 and db is p4 - p3"""

    u = torch.cross(-ab, cb)
    u = u / torch.linalg.norm(u, dim=-1, keepdims=True)
    v = torch.cross(db, cb)
    v = v / torch.linalg.norm(v, dim=-1, keepdims=True)
    w = torch.cross(cb / torch.linalg.norm(cb, dim=-1, keepdims=True), u)

    return torch.stack([(w * v).sum(-1), (u * v).sum(-1)], dim=-1)


def get_bb_torsions(geom):
    c_n = geom[1:, 1] - geom[:-1, 2]
    n_ca = geom[:, 0] - geom[:, 1]
    ca_c = geom[:, 2] - geom[:, 0]

    phi = get_torsions(c_n, n_ca[1:], ca_c[1:])
    psi = get_torsions(n_ca[:-1], ca_c[:-1], c_n)

    return torch.stack([phi, psi], dim=-2)


def bb_torsion_loss(truth, pred):
    true_torsions = get_bb_torsions(truth)
    pred_torsions = get_bb_torsions(pred)

    return (true_torsions - pred_torsions).pow(2).sum(-1).mean()


def clash_check(pred_points, atom14_mask, seq, tolerance=1.5):
    fp_type = pred_points.dtype
    dev = pred_points.device

    atom_radius = torch.tensor([residue_van_der_waals_radius[amino] for amino in seq]).to(dev)
    residue_index = torch.arange(len(seq)).to(device)

    # This function can not handle nans
    pred_points[pred_points != pred_points] = 0.0

    dists = ((pred_points[:, None, :, None, :] - pred_points[None, :, None, :, :]).square().sum(-1) + 1e-8).sqrt()

    # Create the mask for valid distances.
    # shape (N, N, 14, 14)
    dists_mask = atom14_mask[:, None, :, None] * atom14_mask[None, :, None, :]

    # Create mask so we don't double count
    dists_mask = dists_mask * (residue_index[:, None, None, None] < residue_index[None, :, None, None])

    # Contiguous C-N don't count as clashes

    c_one_hot = torch.nn.functional.one_hot(residue_index.new_tensor(2), num_classes=14)
    c_one_hot = c_one_hot.reshape(*((1,) * len(residue_index.shape[:-1])), *c_one_hot.shape)
    c_one_hot = c_one_hot.type(fp_type)

    n_one_hot = torch.nn.functional.one_hot(residue_index.new_tensor(1), num_classes=14)
    n_one_hot = n_one_hot.reshape(*((1,) * len(residue_index.shape[:-1])), *n_one_hot.shape)
    n_one_hot = n_one_hot.type(fp_type)

    neighbour_mask = (residue_index[..., :, None, None, None] + 1) == residue_index[..., None, :, None, None]
    c_n_bonds = neighbour_mask * c_one_hot[..., None, None, :, None] * n_one_hot[..., None, None, None, :]
    dists_mask = dists_mask * (1.0 - c_n_bonds)

    # Disulfide bridge between two cysteines is no clash.

    cys_sg_one_hot = torch.zeros_like(atom14_mask).type(fp_type)
    cys_sg_one_hot[:, 5] = torch.tensor([amino == "C" for amino in seq])

    disulfide_bonds = cys_sg_one_hot[..., None, :, None] * cys_sg_one_hot[..., :, None, :]

    dists_mask = dists_mask * (1.0 - disulfide_bonds)

    accepted_separation = atom_radius[:, None, :, None] + atom_radius[None, :, None, :]

    clashes = (accepted_separation - dists - tolerance).clamp(0)
    clashes[~dists_mask.to(bool)] = 0

    return clashes.sum() / len(seq)


def AF2_generic_fape(rig1, rig2, points1, points2, rigids_mask, mask, *args):
    H_rig1, H_rig2, H_points1, H_points2 = rig1[rigids_mask], rig2[rigids_mask], points1[mask], points2[mask]

    H_on_H_1, H_on_H_2 = (H_rig1.inv() @ vec_from_tensor(H_points1).unsqueeze(-1)), (
            H_rig2.inv() @ vec_from_tensor(H_points2).unsqueeze(-1))

    same_chain_fape = (H_on_H_1.dist(H_on_H_2).flatten() / 10).clamp(max=1.0).mean()

    return  same_chain_fape


def AFM_BB_fape(rig1, rig2, H_mask):
    same_chain_mask = H_mask == H_mask[:, None]

    o1, o2 = rig1.origin.unsqueeze(-1), rig2.origin.unsqueeze(-1)
    x1, x2 = rig1.inv() @ o1, rig2.inv() @ o2

    dists = (x1.dist(x2) / 10)
    same_chain_fape = dists[same_chain_mask].clamp(max=1.0).mean()
    diff_chain_fape = dists[~same_chain_mask].mean()
    return same_chain_fape + diff_chain_fape


def AFM_generic_fape(rig1, rig2, points1, points2, rigids_mask, mask, H_mask, scaling=1):
    H_rig1, H_rig2, H_points1, H_points2 = rig1[H_mask][rigids_mask[H_mask]], rig2[H_mask][rigids_mask[H_mask]], \
                                           points1[H_mask][mask[H_mask]], points2[H_mask][mask[H_mask]]
    L_rig1, L_rig2, L_points1, L_points2 = rig1[~H_mask][rigids_mask[~H_mask]], rig2[~H_mask][rigids_mask[~H_mask]], \
                                           points1[~H_mask][mask[~H_mask]], points2[~H_mask][mask[~H_mask]]

    H_on_H_1, H_on_H_2 = (H_rig1.inv() @ vec_from_tensor(H_points1).unsqueeze(-1)), (
            H_rig2.inv() @ vec_from_tensor(H_points2).unsqueeze(-1))
    L_on_L_1, L_on_L_2 = (L_rig1.inv() @ vec_from_tensor(L_points1).unsqueeze(-1)), (
            L_rig2.inv() @ vec_from_tensor(L_points2).unsqueeze(-1))

    same_chain_fape = (torch.cat([H_on_H_1.dist(H_on_H_2).flatten(), L_on_L_1.dist(L_on_L_2).flatten()]) / 10).clamp(
        max=1.0).mean()

    H_on_L_1, H_on_L_2 = (H_rig1.inv() @ vec_from_tensor(L_points1).unsqueeze(-1)), (
            H_rig2.inv() @ vec_from_tensor(L_points2).unsqueeze(-1))
    L_on_H_1, L_on_H_2 = (L_rig1.inv() @ vec_from_tensor(H_points1).unsqueeze(-1)), (
            L_rig2.inv() @ vec_from_tensor(H_points2).unsqueeze(-1))

    diff_chain_fape = (torch.cat([H_on_L_1.dist(H_on_L_2).flatten(), L_on_H_1.dist(L_on_H_2).flatten()]) / 10).clamp(
        max=3.0).mean()
        

    return diff_chain_fape + same_chain_fape


def get_loss(model, data, j):
    encoding, geom, seq = data.encodings[j], torch.tensor(data.geoms[j]).to(device), data.seqs[j]

    rigid_in, moveable_mask = rigid_in_from_geom_out(data.geoms[j], encoding)
    rigid_true = rigid_from_tensor(geom)
    torsion_true = torsion_angles_from_geom(geom, seq)
    missing_atom_mask = (geom != 0.0).all(-1)

    _, all_reference_frames, all_atoms, all_atom_mask = model(torch.tensor(encoding).to(device), rigid_in.to(device),
                                                              seq, rigid_true=rigid_true, torsion_true=torsion_true,
                                                              update_mask=moveable_mask.to(device))
    mask = missing_atom_mask * all_atom_mask
    loss = model.aux_loss + generic_fape(all_reference_frames, global_frames_from_geom(geom, seq), all_atoms[mask],
                                         geom[mask])
    return loss


def get_loss_fine_tune(model, data, j):
    encoding, geom, seq = data.encodings[j], torch.tensor(data.geoms[j]).to(device), data.seqs[j]

    rigid_in, moveable_mask = rigid_in_from_geom_out(data.geoms[j], encoding)
    rigid_true = rigid_from_tensor(geom)
    torsion_true = torsion_angles_from_geom(geom, seq)
    missing_atom_mask = (geom != 0.0).all(-1)

    _, all_reference_frames, all_atoms, all_atom_mask = model(torch.tensor(encoding).to(device), rigid_in.to(device),
                                                              seq, rigid_true=rigid_true, torsion_true=torsion_true,
                                                              update_mask=moveable_mask.to(device))
    mask = missing_atom_mask * all_atom_mask
    loss = model.aux_loss + generic_fape(all_reference_frames, global_frames_from_geom(geom, seq), all_atoms[mask],
                                         geom[mask])

    return loss + dist_check(all_atoms, geom) + angle_check(all_atoms, geom) + clash_check(all_atoms, mask, seq)


def to_pdb(all_atoms, seq, encoding):
    atom_index = 0
    pdb_lines = []
    record_type = "ATOM"
    chain_ids = "HL"
    chain_index = encoding[:, -1].astype(int)
    chain_start, chain_id = 0, "H"

    for i, amino in enumerate(seq):
        for atom in atom_types:
            if atom in residue_atoms[amino]:
                j = residue_atoms[amino].index(atom)
                pos = all_atoms[i, j]
                name = f' {atom}'
                alt_loc = ''
                res_name_3 = restype_1to3[amino]
                if chain_id != chain_ids[chain_index[i]]:
                    chain_start = i
                    chain_id = chain_ids[chain_index[i]]
                insertion_code = ''
                occupancy = 1.00
                b_factor = 0.00
                element = atom[0]
                charge = ''
                # PDB is a columnar format, every space matters here!
                atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                             f'{res_name_3:>3} {chain_id:>1}'
                             f'{(i + 1 - chain_start):>4}{insertion_code:>1}   '
                             f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                             f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                             f'{element:>2}{charge:>2}')
                pdb_lines.append(atom_line)
                atom_index += 1

    return "\n".join(pdb_lines)
