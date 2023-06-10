import unittest
from RingFinder import *
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, Atom, Bond

class RingFinderTest(unittest.TestCase):

    def test_neighbors1(self):
        smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol)
        neighbors = ringFinder._ring_neighbors(4)
        self.assertCountEqual(neighbors, [3,5])

    def test_neighbors2(self):
        smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol)
        neighbors = ringFinder._ring_neighbors(12)
        self.assertCountEqual(neighbors, [11,13])
    
    def test_neighbors3(self):
        smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol)
        neighbors = ringFinder._ring_neighbors(15)
        self.assertCountEqual(neighbors, [16,24])
    
    def test_neighbors4(self):
        smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol)
        neighbors = ringFinder._ring_neighbors(23)
        self.assertCountEqual(neighbors, [18,24,22])

    def test_rings_no_neighbors(self):
        smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol, False)
        rings = ringFinder.rings
        self.assertCountEqual(rings, [[1,2,3,4,5], [9,10,11,12,13,14],[15,16,17,18,19,20,21,22,23,24]])

    def test_rings_neighbors(self):
        smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol, True)
        rings = ringFinder.rings
        self.assertCountEqual(rings, [[0,1,2,3,4,5,6,25],[8,9,10,11,12,13,14],[15,16,17,18,19,20,21,22,23,24]])
        
    def test_rings_neighbors_lex(self):
        smiles = 'Fc1ccc(cc1)[C@@]3(OCc2cc(C#N)ccc23)CCCN(C)C'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol, True)
        rings = ringFinder.rings
        self.assertCountEqual(rings, [[0,1,2,3,4,5,6,],[7,8,9,10,11,12,13,15,16,17,18]])
    
    def test_branches(self): 
        smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol, True)
        branches = ringFinder.branches
        self.assertCountEqual(branches, [[7]])

    def test_branches_lex(self): 
        smiles = 'Fc1ccc(cc1)[C@@]3(OCc2cc(C#N)ccc23)CCCN(C)C'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol, True)
        branches = ringFinder.branches
        self.assertCountEqual(branches, [[14],[19,20,21,22,23]])

    def test_hydros_lex(self): 
        smiles = 'Fc1ccc(cc1)[C@@]3(OCc2cc(C#N)ccc23)CCCN(C)C'
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        ringFinder = RingFinder(mol, True)
        branches = ringFinder.branches
        self.assertCountEqual(branches, [[14],[19,20,21,22,23,35,36,37,38,39,40,41,42,43,44]])
        rings = ringFinder.rings
        self.assertCountEqual(rings, [[0,1,2,3,4,5,6,24,25,26,27],[7,8,9,10,11,12,13,15,16,17,18,28,29,30,31,32,33,34]])
   

    def test_hydros(self): 
        smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        ringFinder = RingFinder(mol, True)
        rings = ringFinder.rings
        self.assertCountEqual(rings, [[0,1,2,3,4,5,6,25,26,27,28,29,30],[8,9,10,11,12,13,14,31,32,33,34,35,36,37,38,39,40],[15,16,17,18,19,20,21,22,23,24,41,42,43,44,45,46]])
        branches = ringFinder.branches
        self.assertCountEqual(branches, [[7]])
    
       
  


