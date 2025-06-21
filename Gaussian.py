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

def row_matrix_multiplication(A: NDArray, B):
    #specific for 1x3 and 3x3 matrices
    result = [0,0,0]
    for j in range(3):
        for k in range(3):
            result[j] += (A[0][j] * B[k][j] )
    return np.array(result) 

def matrix_multiplication_column(A, B):
     #specific for 3 x 3 and 3 x 1 matrices
    result = [0, 0, 0]
    for i in range(3):
        for k in range(3):
            result[i] += (B[k][0] * A[k][i])
    return np.array(result)

def matrix_multiplication_row_column(A,B):
    result = [0]
    for k in range(3):
        result[0] += A[0][k] * B[k][0]
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
    matrixA: NDArray

    def __init__(self, covariance_inverse_matrix: np.ndarray, center: np.ndarray, matrixA: np.ndarray) -> None:
        self.convariance_matrix_inverse = covariance_inverse_matrix
        self.covariance_matrix = inverse_of_matrix(covariance_inverse_matrix)
        self.center = center
        ellipse = quadratic_to_parametric(center, covariance_inverse_matrix)
        self.a = ellipse.axes[0]
        self.b = ellipse.axes[1]
        self.c = ellipse.axes[2]
        self.matrixA = matrixA

    

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

        return cls(matrixA, center, matrixA)
    
    def volume_constant(self):
        det = deteriminant_of_matrix(self.convariance_matrix_inverse)
        volume = ellipse_volume(self.a , self.b, self.c)
        N = (det / (np.pi **3))** 0.5 * volume
        return N
    
    def grid_volume(self):
        # Create a grid of points in 3D space
        lengtha = np.linalg.norm(self.a)
        lengthb = np.linalg.norm(self.b)
        lengthc = np.linalg.norm(self.c)
        lengths = [lengtha, lengthb, lengthc]
        longest_item = 0
        second_longest_item = 0
        short_item = 0
        for length in lengths:
            if length > int(longest_item):
                second_longest_item = longest_item
                longest_item = length
        x = np.linspace(self.center[0]-longest_item, self.center[0]+ longest_item, 200)
        y = np.linspace(self.center[1]-longest_item, self.center[1]+ longest_item, 200)
        z = np.linspace(self.center[2]-longest_item, self.center[2]+ longest_item, 200)
        points_in_ellipse = 0 
        gaussian = Gaussian.from_axes(self.a, self.b, self.c, self.center)
        matrixA = gaussian.matrixA
        for i in x:
                for j in y:
                        for k in z:
                                point = np.array([[i - self.center[0]], [j - self.center[1]], [k - self.center[2]]])
                                transpose_r = np.transpose(point)
                                XTG = row_matrix_multiplication(transpose_r, matrixA)
                                value = matrix_multiplication_row_column([XTG], point)
                                if value < 1:
                                    points_in_ellipse += 1
        dx = 2 * longest_item / 200
        dy = 2 * longest_item / 200
        dz = 2 * longest_item / 200
        point_volume = dx * dy * dz
        ellipse_volume = points_in_ellipse * point_volume
        return ellipse_volume
    

    



                    
    
    
