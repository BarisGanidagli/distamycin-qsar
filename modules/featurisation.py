"""Morgan fingerprints and physicochemical descriptors from SMILES."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs

_MORGAN_RADIUS = 2
_MORGAN_BITS = 1024
_DESC_NAMES = ['MolWt', 'LogP', 'HDonors', 'HAcceptors', 'RotBonds', 'TPSA']


def smiles_to_fp_desc(smiles):
    """Convert SMILES to a Morgan fingerprint and physicochemical descriptors."""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(_MORGAN_BITS), np.zeros(len(_DESC_NAMES))

    generator = AllChem.GetMorganGenerator(radius=_MORGAN_RADIUS, fpSize=_MORGAN_BITS)
    fp = np.zeros(_MORGAN_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), fp)

    desc = np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.TPSA(mol),
    ])

    return fp, desc


def featurise_molecules(smiles_list):
    """Featurise a list of SMILES into fingerprints and descriptors."""

    rows = []
    for smi in smiles_list:
        fp, desc = smiles_to_fp_desc(smi)
        rows.append(np.concatenate([fp, desc]))
    return np.vstack(rows)


def get_morgan_fps(smiles_series, radius=_MORGAN_RADIUS, n_bits=_MORGAN_BITS):
    """Return RDKit Morgan bit-vector fingerprints."""

    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return [generator.GetFingerprint(Chem.MolFromSmiles(smi)) for smi in smiles_series]