import unittest
from Molecule import *
from numpy.linalg import LinAlgError

class MoleculeTest(unittest.TestCase):

    def test_ethane_with_hydrogens(self):
        smiles = 'CC'
        expandAtom = False
        includeHydros = True
        programInput = ProgramInput(includeHydros, expandAtom, smiles)
        output = find_ellipses(programInput)
        self.assertEqual(1, len(output.ellipsis))
        ellipse = output.ellipsis[0]
        magnitudes = ellipse.axes_magnitudes
        self.assertAlmostEqual(1.254, magnitudes[0], 2)
        self.assertAlmostEqual(1.254, magnitudes[1], 2)
        self.assertAlmostEqual(1.975, magnitudes[2], 2)
        number_atoms = output.mol.GetNumAtoms();
        self.assertEqual(number_atoms, len(ellipse.points))

    def test_ethane_with_expandAtom(self):
        smiles = 'CC'
        expandAtom = True
        includeHydros = True
        programInput = ProgramInput(includeHydros, expandAtom, smiles)
        output = find_ellipses(programInput)
        self.assertEqual(1, len(output.ellipsis))
        ellipse = output.ellipsis[0]
        magnitudes = ellipse.axes_magnitudes
        self.assertAlmostEqual(2.278, magnitudes[0], 1)
        self.assertAlmostEqual(2.435, magnitudes[1], 1)
        self.assertAlmostEqual(2.723, magnitudes[2], 1)
        number_atoms = output.mol.GetNumAtoms();
        self.assertEqual(6*number_atoms, len(ellipse.points))
        
    def test_benzene_with_hydrogens(self):
        smiles = "c1ccccc1"
        expandAtom = False
        includeHydros = True
        programInput = ProgramInput(includeHydros, expandAtom, smiles)
        output = find_ellipses(programInput)
        self.assertEqual(1, len(output.ellipsis))
        ellipse = output.ellipsis[0]
        magnitudes = ellipse.axes_magnitudes
        self.assertAlmostEqual(1.257e-05, magnitudes[0], 3)
        self.assertAlmostEqual(2.65, magnitudes[1], 1)
        self.assertAlmostEqual(2.731, magnitudes[2], 1)
        number_atoms = output.mol.GetNumAtoms();
        self.assertEqual(number_atoms, len(ellipse.points))
        
    def test_benzene_without_hydrogens(self):
        smiles = "c1ccccc1"
        expandAtom = False
        includeHydros = False
        programInput = ProgramInput(includeHydros, expandAtom, smiles)
        output = find_ellipses(programInput)
        self.assertEqual(1, len(output.ellipsis))
        ellipse = output.ellipsis[0]
        magnitudes = ellipse.axes_magnitudes
        self.assertAlmostEqual(1.257e-05, magnitudes[0], 3)
        self.assertAlmostEqual(1.482, magnitudes[1], 1)
        self.assertAlmostEqual(1.899, magnitudes[2], 1)
        number_atoms = output.mol.GetNumAtoms();
        self.assertEqual(number_atoms - 6, len(ellipse.points))
    
    def test_hydrogen_cyanide_with_hydrogens(self):
        smiles = "C # N"
        expandAtom = False
        includeHydros = True
        programInput = ProgramInput(includeHydros, expandAtom, smiles)
        output = find_ellipses(programInput)
        self.assertEqual(1, len(output.ellipsis))
        ellipse = output.ellipsis[0]
        magnitudes = ellipse.axes_magnitudes
        self.assertAlmostEqual(1.092, magnitudes[0], 1)
        self.assertAlmostEqual(1.092, magnitudes[1], 1)
        self.assertAlmostEqual(1.092, magnitudes[2], 1)
        number_atoms = output.mol.GetNumAtoms();
        self.assertEqual(number_atoms, len(ellipse.points))


    def test_ethane_without_hydrogens(self):
        smiles = 'CC'
        expandAtom = False
        includeHydros = False
        programInput = ProgramInput(includeHydros, expandAtom, smiles)
        with self.assertRaises(LinAlgError):
            find_ellipses(programInput)


if __name__ == "__main__":
    unittest.main()

