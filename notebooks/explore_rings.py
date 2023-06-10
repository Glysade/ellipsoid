# %%

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdRGroupDecomposition import RGroupDecomposition, RGroupDecompositionParameters, \
    RGroupMatching, RGroupScore, RGroupLabels, RGroupCoreAlignment, RGroupLabelling
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.MolStandardize import rdMolStandardize;
from rdkit.Chem.rdchem import Mol, Atom, Bond

IPythonConsole.ipython_useSVG = True
IPythonConsole.drawOptions.addAtomIndices = True
# IPythonConsole.drawOptions.addBondIndices = True
IPythonConsole.molSize = (800, 500)

from rdkit.Chem import PandasTools
from rdkit.Chem import Draw
from IPython.display import HTML

import pandas as pd
# %%
smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
mol: Mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
mol

# %%

atom: Atom
for atom in mol.GetAtoms():
    in_ring = atom.IsInRing()
    atom_idx = atom.GetIdx()
    print(f'atom {atom_idx} is in ring {in_ring}')

# %%

bond:Bond
for bond in mol.GetBonds():
    in_ring = bond.IsInRing()
    bond_idx = bond.GetIdx()
    begin_atom_idx = bond.GetBeginAtomIdx()
    end_atom_idx = bond.GetEndAtomIdx()
    print(f'bond {bond_idx} is in ring {in_ring} atom1 {begin_atom_idx} atom2 {end_atom_idx}')
bond:Bond


# %%

bond: Bond = mol.GetBondBetweenAtoms(5, 1)
bond.GetIdx()
# %%

atom_idx = 1
atom: Atom = mol.GetAtomWithIdx(atom_idx)
neighbors = atom.GetNeighbors()
neighbors
# = anything the atom is bonded to


# %%
neighbor: Atom
for neighbor in neighbors:
    idx = neighbor.GetIdx()
    bond: Bond = mol.GetBondBetweenAtoms(atom_idx, idx)
    bond_idx = bond.GetIdx()
    print(f'Neighbor of Atom {atom_idx} has atom index {idx} bond between has index {bond_idx}')

# %%

smiles = 'Fc1ccc(cc1)[C@@]3(OCc2cc(C#N)ccc23)CCCN(C)C'
mol: Mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
mol
# %%
