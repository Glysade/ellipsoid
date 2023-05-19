from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
import numpy as np
import numpy.linalg as la
from numpy.typing import NDArray
from typing import NamedTuple
import os
import sys
from dataclasses import dataclass
import argparse

@dataclass
class Ellipse:
    center: NDArray
    square_matrix: NDArray
    eigen_values: NDArray
    eigen_vectors: NDArray
    axes_magnitudes: NDArray
    axes: NDArray

def generate_3d_conformation(mol: Mol) -> Mol:
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol

def molecule_points(mol: Mol, includeHydro, expandAtom) -> NDArray:
    conformer = mol.GetConformer()
    coordinates = []
    for atom in mol.GetAtoms():
        x = atom.GetAtomicNum()
        if includeHydro or atom.GetAtomicNum() > 1:
            index = atom.GetIdx()
            if expandAtom:
                coordinate = conformer.GetAtomPosition(index)
                r = Chem.GetPeriodicTable().GetRvdw(x)
                coordinate.x = coordinate.x + r;
                coordinates.append(coordinate)
                coordinate = conformer.GetAtomPosition(index)
                coordinate.x = coordinate.x - r;
                coordinates.append(coordinate)
                coordinate = conformer.GetAtomPosition(index)
                coordinate.y = coordinate.y + r;
                coordinates.append(coordinate)
                coordinate = conformer.GetAtomPosition(index)
                coordinate.y = coordinate.y - r;
                coordinates.append(coordinate)
                coordinate = conformer.GetAtomPosition(index)
                coordinate.z = coordinate.z + r;
                coordinates.append(coordinate)
                coordinate = conformer.GetAtomPosition(index)
                coordinate.z = coordinate.z - r;
                coordinates.append(coordinate)
            else:
                coordinate = conformer.GetAtomPosition(index)
                coordinates.append(coordinate)

    return np.asarray(coordinates)


# from https://gist.github.com/Gabriel-p/4ddd31422a88e7cdf953
def mvee(points, tol=0.0001) -> [NDArray,NDArray]:
    """
    Finds the ellipse equation in "center form"
    (x-c).T * A * (x-c) = 1
    """
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        M = np.diag(np.dot(np.dot(Q.T, la.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = la.norm(new_u-u)
        u = new_u
    c = np.dot(u, points)
    A = la.inv(np.dot(np.dot(points.T, np.diag(u)), points)
               - np.multiply.outer(c, c))/d
    return A, c
    
def quadratic_to_parametric(center: NDArray, A: NDArray) -> Ellipse:
    # eigenvalues and eigenvectors using SVD
    # see https://en.wikipedia.org/wiki/Singular_value_decomposition
    # and https://laurentlessard.com/teaching/cs524/slides/11%20-%20quadratic%20forms%20and%20ellipsoids.pdf
    # For square symmetric A = U.D.VT matrix U = V 
    # and Eigen values of A are singular values in D
    # Rows of U are Eigen vectors of A
    U, D, VT = la.svd(A)
    axes_magnitudes = 1.0/np.sqrt(D)
    # hack to multiply by row instead of column
    axes = U * axes_magnitudes[:, np.newaxis]

    ellipse = Ellipse(center = center, square_matrix = A, eigen_values = D, eigen_vectors = U, axes_magnitudes = axes_magnitudes, axes = axes)
    return ellipse


def print_pymol_ellipse(mol: Mol, base: str, ellipse: Ellipse) -> None:

    block = Chem.MolToMolBlock(mol)
    out_file = f'{base}.sdf'
    with open(out_file, 'wt') as fh:
        fh.write(block)
    full_sd_path = os.getcwd() + os.path.sep + out_file;

    center = ellipse.center
    mag = ellipse.axes_magnitudes
    rot = ellipse.eigen_vectors
    drawCommand = f'tmp = drawEllipsoid([0.85, 0.85, 1.00] '
    for i in range(3):
        drawCommand = drawCommand + f', {center[i]}'
    for i in range(3):
        drawCommand = drawCommand + f', {mag[i]}'
    for i in range(3):
        for j in range(3):
            drawCommand = drawCommand + f', {rot[i][j]}'
    drawCommand = drawCommand + ')'
    py_script = f'{base}.py'
    with open(py_script, 'wt') as fh:
        fh.write("cmd.delete('all')\n")
        fh.write(f"cmd.load('{full_sd_path}')\n")
        fh.write(drawCommand)
        fh.write('\n')
        fh.write("cmd.load_cgo(tmp, 'ellipsoid-cgo')\n")
        fh.write("cmd.set('cgo_transparency', 0.5, 'ellipsoid-cgo')\n")
    
    full_py_path = os.getcwd() + os.path.sep + py_script
    print(f'Pymol script {full_py_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles")
    parser.add_argument("--includeHydros", action=argparse.BooleanOptionalAction)
    parser.add_argument("--expandAtom", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    print(f'Smiles from arguments is {args.smiles}')
    print(f'includeHydros from arguments is {args.includeHydros}')
    print(f'expandAtom from arguments is {args.expandAtom}')
    smiles = args.smiles
    if not smiles:
        print('Using default smiles')
        smiles = "CC(C)C[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CO)NC(=O)[C@@H]1CCCN1C(=O)[C@H](CCC(N)=O)NC(=O)[C@@H](N)CS)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CS)C(=O)O"
    includeHydros = args.includeHydros
    expandAtom = args.expandAtom
    mol = Chem.MolFromSmiles(smiles)
    mol = generate_3d_conformation(mol)
    # find points on ellipsoid
    points = molecule_points(mol, includeHydros, expandAtom)
    # find MVEE
    # Quadratic form defined by square symmetric matrix A and centered at centroid
    A, center = mvee(points);

    ellipsoid = quadratic_to_parametric(center, A)
    print_pymol_ellipse(mol, 'out', ellipsoid)    




