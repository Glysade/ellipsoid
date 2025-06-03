# Molecule Command
This is a description of the molecule.py page and included functions. 
## Packages
- **RDKIT** is a library for cheminfomatics
- **NumPy** is a package for working with arrays and linear algebra
## Functions
### Generate 3d conformation
Smiles does not explicitly include the hydrogen atoms in the string, so we use addHs to include the Hydrogens in the molecule where there are available endpoints. 

` mol = Chem.AddHs(mol) AllChem.EmbedMolecule(mol) AllChem.MMFFOptimizeMolecule(mol) ` 
### Molecule Points
Create an empty list for both coordinates and atoms. 

Use mol.GetAtoms to find atoms from the molecule. Use GetAtomWithIdx append atom  index to list
### Maximum Volume Enclosing Ellipsoid (MVEE)
### Quadratic to Parametric
### Print
### Find Ellipses

### Main Commands
- smiles - imput the smiles for the desired compound
we will extract the molecule using smiles from Chem 
`  if '\n' in programInput.smiles:
        mol = Chem.MolFromMolBlock(programInput.smiles) ` 
if smiles is not provided we will use the default smiles. 
` if not smiles:
        print('Using default smiles')
        smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br' ` 
- expand atom. If selected as true the atoms will be represented as a series of points, not a single point. There is a coordinate bias. The molecule will appear more distored at certain angles of rotation. 
` if expandAtom:
                coordinate = conformer.GetAtomPosition(index)
                r = Chem.GetPeriodicTable().GetRvdw(x)
                coordinate.x = coordinate.x + r;
                coordinates.append(coordinate)
                coordinate = conformer.GetAtomPosition(index)
                coordinate.x = coordinate.x - r; ` 
- fragment: if fragment is True fragments are created, which are a list of lists, comprised of rings and branches. Then create an ellipsoid around the points of the fragment, so that each fragment is represented by its own ellipsoid. 
- Number Neighbors: Number neighbor arg goes to ringfinder. 
In the find complete rings function, a for loop runs in range number neighbors. It appends atoms in range, gets their neighbors, and appends them also. 
` atom_index = ring[i]
                    atom = self.mol.GetAtomWithIdx(atom_index)
                    neighbors = atom.GetNeighbors() ` 
- Merge Length: also runs in ringfinder 
`if len(branch) <= mergeLength:` program loops through the length of the branch. Within range it finds atoms and neighbors. If the index of a neighbor is equal to the index in another branch, the two branches will be merged. 
- prints given smiles and expand atom. 

