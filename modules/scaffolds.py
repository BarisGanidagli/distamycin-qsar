"""Murcko scaffolds and scaffold-based splitting of molecular data."""

import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_scaffold(smiles):
    """Return the Murcko scaffold for a SMILES string, or None if it cannot be parsed."""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.GetScaffoldForMol(mol)


def scaffold_split(df, n_splits=5, random_state=42):
    """Split a molecular dataset into scaffold-based folds."""

    scaffolds = defaultdict(list)
    for idx, smi in enumerate(df['SMILES']):
        scaffold = get_scaffold(smi)
        key = Chem.MolToSmiles(scaffold) if scaffold is not None else None
        scaffolds[key].append(idx)

    rng = np.random.default_rng(random_state)
    scaffold_keys = list(scaffolds.keys())
    rng.shuffle(scaffold_keys)
    scaffold_keys.sort(key=lambda k: len(scaffolds[k]), reverse=True)

    folds = [[] for _ in range(n_splits)]
    for key in scaffold_keys:
        smallest = min(range(n_splits), key=lambda f: len(folds[f]))
        folds[smallest].extend(scaffolds[key])

    return folds