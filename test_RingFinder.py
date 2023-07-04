import unittest
from RingFinder import *
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, Atom, Bond

class RingFinderTest(unittest.TestCase):

    def test_neighbors1(self):
        smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol,1,1)
        neighbors = ringFinder._ring_neighbors(4)
        self.assertCountEqual(neighbors, [3,5])

    def test_neighbors2(self):
        smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol,1,1)
        neighbors = ringFinder._ring_neighbors(12)
        self.assertCountEqual(neighbors, [11,13])
    
    def test_neighbors3(self):
        smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol,1,2)
        neighbors = ringFinder._ring_neighbors(15)
        self.assertCountEqual(neighbors, [16,24])
    
    def test_neighbors4(self):
        smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol,1,1)
        neighbors = ringFinder._ring_neighbors(23)
        self.assertCountEqual(neighbors, [18,24,22])

    def test_rings_no_neighbors(self):
        smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol, 0, 2)
        rings = ringFinder.rings
        self.assertCountEqual(rings, [[1,2,3,4,5], [9,10,11,12,13,14],[15,16,17,18,19,20,21,22,23,24]])

    def test_rings_neighbors(self):
        smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol, 1, 2)
        rings = ringFinder.rings
        self.assertCountEqual(rings, [[0,1,2,3,4,5,6,25],[8,9,10,11,12,13,14],[15,16,17,18,19,20,21,22,23,24]])
        
    def test_rings_neighbors_lex(self):
        smiles = 'Fc1ccc(cc1)[C@@]3(OCc2cc(C#N)ccc23)CCCN(C)C'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol, 1, 2)
        rings = ringFinder.rings
        self.assertCountEqual(rings, [[0,1,2,3,4,5,6,],[7,8,9,10,11,12,13,15,16,17,18]])

    def test_branches_lex(self): 
        smiles = 'Fc1ccc(cc1)[C@@]3(OCc2cc(C#N)ccc23)CCCN(C)C'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol, 2, 2)
        branches = ringFinder.branches
        self.assertCountEqual(branches, [[20,21,22,23]])


    def test_hydros(self): 
        smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol, 1, 2)
        rings = ringFinder.rings
        self.assertCountEqual(rings, [[0,1,2,3,4,5,6,25],[8,9,10,11,12,13,14,],[15,16,17,18,19,20,21,22,23,24]])
        branches = ringFinder.branches
        self.assertCountEqual(branches, [[7]])
    
    def test_neigh(self): 
        smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol, 0, 2)
        rings = ringFinder.rings
        self.assertCountEqual(rings, [[1,2,3,4,5],[9,10,11,12,13,14],[15,16,17,18,19,20,21,22,23,24]])
        branches = ringFinder.branches
        self.assertCountEqual(branches, [[0],[6,7,8],[25]])
    

    def test_new_smiles(self): 
        smiles = 'CCCCC(=O)N(CC1=CC=C(C=C1)C2=CC=CC=C2C3=NN=N[N-]3)C(C(C)C)C(=O)[O-].CCCCC(=O)N(CC1=CC=C(C=C1)C2=CC=CC=C2C3=NN=N[N-]3)C(C(C)C)C(=O)[O-].CCOC(=O)C(C)CC(CC1=CC=C(C=C1)C2=CC=CC=C2)NC(=O)CCC(=O)[O-].CCOC(=O)C(C)CC(CC1=CC=C(C=C1)C2=CC=CC=C2)NC(=O)CCC(=O)[O-].O.O.O.O.O.[Na+].[Na+].[Na+].[Na+].[Na+].[Na+]'
        mol = Chem.MolFromSmiles(smiles)
        ringFinder = RingFinder(mol, 1, 2)
        rings = ringFinder.rings
        self.assertCountEqual(rings, [[7,8,9,10,11,12,13],[14,15,16,17,18,19],[20,21,22,23,24],[39,40,41,42,43,44,45],[46,47,48,49,50,51],[52,53,54,55,56],[73,74,75,76,77,78,79],[80,81,82,83,84,85],[103,104,105,106,107,108,109],[110,111,112,113,114,115]])
        branches = ringFinder.branches
        self.assertCountEqual(branches, [[0,1,2],[3,4,5,6],[25,26,27,28,29,30,31],[32,33,34],[35,36,37,38],[57,58,59,60,61,62,63],[64,65,66,67,68],[69,70,71,72],[86,87,88,89],[90,91,92,93],[94,95,96,97,98],[99,100,101,102],[116,117,118,119],[120,121,122,123]])
       
  


