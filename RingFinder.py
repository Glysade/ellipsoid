


from rdkit import Chem
from rdkit.Chem.rdchem import Mol, Atom, Bond

class RingFinder:
    def __init__(self, mol, numberNeighbors):
        self.mol = mol 
        assigned_atoms = set()
        self.assigned_atoms = assigned_atoms
        self.numberNeighbors = numberNeighbors

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
        self._find_rings()
        self._find_complete_rings(numberNeighbors)
        self._branch()
        self._merge_branches()
        
    def _ring_neighbors(self, atom_index):
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

    def _find_next_ring(self):
        ring_atoms = self.ring_atoms
        assigned_atoms = self.assigned_atoms
        start = None
        for ring_atom in ring_atoms:
            if ring_atom not in assigned_atoms:
                start = ring_atom
                break
        if start is None:
            return []
        current_ring = [start]   
        assigned_atoms.add(start)
        grow_ring = True
        while grow_ring:
            grow_ring = False
            for i in range(len(current_ring)):
                current_atom = current_ring[i]
                neighbors = self._ring_neighbors(current_atom)
                for neighbor in neighbors:
                    if neighbor not in current_ring:
                        grow_ring = True
                        current_ring.append(neighbor)
                        assigned_atoms.add(neighbor)
        current_ring.sort()
        return current_ring

    def _find_rings(self):
        all_rings = []
        current_ring = self._find_next_ring()
        while current_ring:
            all_rings.append(current_ring)
            current_ring = self._find_next_ring()
        self.rings = all_rings

    def _find_complete_rings(self, numberNeighbors):
        for _ in range(numberNeighbors):
            for ring in self.rings:
                for i in range(len(ring)):
                    atom_index = ring[i]
                    atom = self.mol.GetAtomWithIdx(atom_index)
                    neighbors = atom.GetNeighbors()
                    for neighbor in neighbors:
                        idx = neighbor.GetIdx()
                        if idx not in self.assigned_atoms:
                            ring.append(idx)
                            self.assigned_atoms.add(idx)
                ring.sort()
   
    def _branch(self):
        degree = []
        branch = []
        branches = []
        self.branches = branches
        for atom in self.mol.GetAtoms():
            atom_idx = atom.GetIdx();
            if atom_idx not in self.assigned_atoms:
                degree = atom.GetDegree()
                if degree == 1:
                    self.assigned_atoms.add(atom_idx)
                    branches.append([atom_idx])
        grow_branch = True
        while grow_branch:
            grow_branch = False
            for branch in branches:
                for i in range(len(branch)):
                        atom = self.mol.GetAtomWithIdx(branch[i])
                        neighbors = atom.GetNeighbors()
                        for neighbor in neighbors:
                                idx = neighbor.GetIdx()
                                if idx not in self.assigned_atoms:
                                    branch.append(idx)
                                    self.assigned_atoms.add(idx)
                                    grow_branch = True
        branches.sort()
        self.branch = branch
        self.branches = branches
        return branches
   
    def _merge_branches(self):
        for z in range(len(self.branches)):
            branch = self.branches[z]
            if len(branch) < 2:
                match = False
                for i in range(len(branch)):
                    atom = self.mol.GetAtomWithIdx(branch[i])
                    neighbors = atom.GetNeighbors()
                    for neighbor in neighbors:
                        idx = neighbor.GetIdx()
                        for j in range(len(self.branches)):
                            #2 for loops
                            #j and z not equal
                            if j == z:
                                continue
                            
                            other_branch = self.branches[j]
                            for other_idx in other_branch:
                                if other_idx == idx:
                                    match = True
                                    break
                            if match:
                                other_branch.extend(branch)
                                branch.clear()
                                other_branch.sort()
                                break
                    if match:
                        break
                if match:
                    break
        old_branches = self.branches
        self.branches = []
        for old_branch in old_branches:
            if old_branch != []: 
                    self.branches.append(old_branch)
                                   

                                

if __name__ == '__main__':
    smiles = 'Fc1ccc(cc1)[C@@]3(OCc2cc(C#N)ccc23)CCCN(C)C'
    m = Chem.MolFromSmiles(smiles)
    ringFinder = RingFinder(m, 2)
    pass

#Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br
#Fc1ccc(cc1)[C@@]3(OCc2cc(C#N)ccc23)CCCN(C)C