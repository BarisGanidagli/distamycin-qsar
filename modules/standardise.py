"""Canonical SMILES standardisation with optional desalting and neutralisation."""

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

_fragment_parent = rdMolStandardize.FragmentParent
_uncharger = rdMolStandardize.Uncharger()

def canonical_smiles(smiles, desalt=False, neutralise=False):
    """Canonical SMILES for the input, or None if it cannot be parsed."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    if desalt:
        mol = _fragment_parent(mol)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
    if neutralise:
        mol = _uncharger.uncharge(mol)
    return Chem.MolToSmiles(mol)
