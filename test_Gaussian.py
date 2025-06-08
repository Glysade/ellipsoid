import numpy as np 
import unittest
from Gaussian import *
class TestGaussian(unittest.TestCase): 
    def test_deteriminant_of_matrix(self):
        matrix = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
        expected_determinant = np.linalg.det(matrix)
        determinantofmatrix = deteriminant_of_matrix(matrix)
        self.assertAlmostEqual(expected_determinant,determinantofmatrix)
    
    def test_deteriminant_of_matrix_mol(self):
        matrix2 = np.array([[ 1.65932389e-01,  2.94332300e-03, -6.81848513e-03],
       [ 2.94332300e-03,  3.13358678e-01,  4.60974143e-05],
       [-6.81848513e-03,  4.60974143e-05,  3.13354109e-01]])
        expected_determinant = np.linalg.det(matrix2)
        determinantofmatrix = deteriminant_of_matrix(matrix2)
        self.assertAlmostEqual(expected_determinant,determinantofmatrix)

    def test_inverse_of_matrix(self):
        matrix = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
        expected_inverse = np.linalg.inv(matrix)
        inverse = inverse_of_matrix(matrix)
        check = np.allclose(expected_inverse, inverse)
        self.assertTrue(check)

    '''def test_ellipse_volume(self):
        volume = ellipse_volume()
        expected_volume = (4.0 / 3.0) * np.pi * (1) * (1) * (1)
        self.assertAlmostEqual(volume, expected_volume)'''
    
    def test_matrix_multiplication(self):
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
        expected_product = np.array([[  30,  24,  18],
                                   [  84,  69,  54],
                                   [138, 114, 90]])
        product = matrix_multiplication(A.copy(), B.copy())
        check = np.allclose(expected_product, product)
        self.assertTrue(check)
        return np.dot(A, B)


    