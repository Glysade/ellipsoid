from typing import List
import numpy as np
from numpy.typing import NDArray

from Molecule import quadratic_to_parametric

def deteriminant_of_matrix(matrix):
    #det(M) = \(a(ei-fh)-b(di-fg)+c(dh-eg)\)
    determinant = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) \
                 - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) \
                    + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    return determinant

def inverse_of_matrix(matrix):
    # Inverse of a 3x3 matrix
    determinant = deteriminant_of_matrix(matrix)
    if determinant == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")
    cofactor_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    cofactor_matrix[0][0] = (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) 
    cofactor_matrix[0][1] = -(matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
    cofactor_matrix[0][2] = (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) 
    cofactor_matrix[1][0] = -(matrix[0][1] * matrix[2][2] - matrix[0][2] * matrix[2][1])    
    cofactor_matrix[1][1] = (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0])
    cofactor_matrix[1][2] = -(matrix[0][0] * matrix[2][1] - matrix[0][1] * matrix[2][0])
    cofactor_matrix[2][0] = (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1])
    cofactor_matrix[2][1] = -(matrix[0][0] * matrix[1][2] - matrix[0][2] * matrix[1][0])
    cofactor_matrix[2][2] = (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])

    transpose_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    transpose_matrix[0][0] = cofactor_matrix[0][0] 
    transpose_matrix[0][1] = cofactor_matrix[1][0] 
    transpose_matrix[0][2] = cofactor_matrix[2][0] 
    transpose_matrix[1][0] = cofactor_matrix[0][1] 
    transpose_matrix[1][1] = cofactor_matrix[1][1] 
    transpose_matrix[1][2] = cofactor_matrix[2][1] 
    transpose_matrix[2][0] = cofactor_matrix[0][2] 
    transpose_matrix[2][1] = cofactor_matrix[1][2] 
    transpose_matrix[2][2] = cofactor_matrix[2][2] 
    inv_matrix = transpose_matrix / determinant
    return inv_matrix

def matrix_multiplication(A, B):
    #specific for 3x3 matrices
    result =[[0,0,0], [0,0,0],[0,0,0]]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += (A[i][k] * B[k][j] )
    return np.array(result)

def ellipse_volume(a, b, c,):
        lengtha = np.linalg.norm(a)
        lengthb = np.linalg.norm(b)
        lengthc = np.linalg.norm(c)
        volume = (4.0 / 3.0) * np.pi * lengtha * lengthb * lengthc
        return volume


class Gaussian:
    covariance_matrix: NDArray
    convariance_matrix_inverse: NDArray
    center: NDArray
    a: NDArray
    b: NDArray
    c: NDArray

    def __init__(self, covariance_inverse_matrix: np.ndarray, center: np.ndarray):
        self.convariance_matrix_inverse = covariance_inverse_matrix
        self.covariance_matrix = inverse_of_matrix(covariance_inverse_matrix)
        self.center = center
        ellipse = quadratic_to_parametric(center, covariance_inverse_matrix)
        self.a = ellipse.axes[0]
        self.b = ellipse.axes[1]
        self.c = ellipse.axes[2]

    

    @classmethod
    def from_axes(cls, a, b, c, center):
        """
        Create a Gaussian from the axes and center.
        """
        A = np.dot(a, a)
        B = np.dot(b, b)
        C = np.dot(c, c)

        # create the inverse of covariance matrix from the axes
        a, b, c = np.array(a), np.array(b), np.array(c)
        u1 = a / (A ** 0.5) # type: ignore
        u2 = b / (B ** 0.5)
        u3 = c / (C ** 0.5)
        d = 1 / (np.linalg.norm(a))**2
        e = 1 / (np.linalg.norm(b))**2
        f = 1 / (np.linalg.norm(c))**2
        matrixD = np.diag([d, e, f])
        transposeU = np.array([u1, u2, u3])
        matrixU = np.matrix.transpose(transposeU) # type: ignore
        #matric A = U * D * U^T
        UD = matrix_multiplication(matrixU, matrixD)
        matrixA = matrix_multiplication(UD, transposeU)

        return Gaussian(matrixA, center)  
    
    def volume_constant(self, a, b, c, center):
        det = deteriminant_of_matrix(self.convariance_matrix_inverse)
        volume = ellipse_volume(a , b, c)
        N = (det / (np.pi **3))** 0.5 * volume
        return N
    
    
