


from rdkit import Chem
from rdkit.Chem.rdchem import Mol, Atom, Bond

class RingFinder:
    def __init__(self, mol):
        self.mol = mol 

        #find atoms in rings 
        ring_atoms = set()
        for atom in mol.GetAtoms():
            if atom.IsInRing() == True: 
                i = atom.GetIdx()
                ring_atoms.add(i)
        self.ring_atoms = ring_atoms
        
        ring_bonds = set()
        for bond in mol.GetBonds():
            if bond.IsInRing() == True:
                j = bond.GetIdx()
                ring_bonds.add(j)
        self.ring_bonds = ring_bonds

    def ring_neighbors(self, atom_index):
        ring_neighbors = []
        if atom_index not in self.ring_atoms:
            return []
        atom = self.mol.GetAtomWithIdx(atom_index)
        neighbors = atom.GetNeighbors()
        for neighbor in neighbors:
            idx = neighbor.GetIdx()
            bond: bond = self.mol.GetBondBetweenAtoms(atom_index, idx)
            if bond.IsInRing() and neighbor.IsInRing():
                ring_neighbors.append(idx)
        return ring_neighbors
    
    

if __name__ == '__main__':
    smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
    m = Chem.MolFromSmiles(smiles)
    ringFinder = RingFinder(m)
    neighbors = ringFinder.ring_neighbors(4)
    # shoud be [5, 3]


