import unittest
from RingFinder import *
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, Atom, Bond

class RingFinderTest(unittest.TestCase):

    def test_neighbors1(self):
        smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol)
        neighbors = ringFinder.ring_neighbors(4)
        self.assertCountEqual(neighbors, [3,5])

    def test_neighbors2(self):
        smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol)
        neighbors = ringFinder.ring_neighbors(12)
        self.assertCountEqual(neighbors, [11,13])
    
    def test_neighbors3(self):
        smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol)
        neighbors = ringFinder.ring_neighbors(15)
        self.assertCountEqual(neighbors, [16,24])
    
    def test_neighbors4(self):
        smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol)
        neighbors = ringFinder.ring_neighbors(23)
        self.assertCountEqual(neighbors, [18,24,22])
