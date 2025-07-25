import math
from typing import List
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from Molecule import quadratic_to_parametric
import numpy.linalg as la

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
    n: float


    def __init__(self, covariance_inverse_matrix: np.ndarray, center: np.ndarray) -> None:
        self.convariance_matrix_inverse = covariance_inverse_matrix
        self.covariance_matrix = inverse_of_matrix(covariance_inverse_matrix)
        self.center = center
        ellipse = quadratic_to_parametric(center, covariance_inverse_matrix)
        self.a = ellipse.axes[0]
        self.b = ellipse.axes[1]
        self.c = ellipse.axes[2]
        self.matrixA = covariance_inverse_matrix
        self.n = 2.418

    

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

        return cls(matrixA, center)
    
    def volume_constant(self):
        det = deteriminant_of_matrix(self.convariance_matrix_inverse)
        volume = ellipse_volume(self.a , self.b, self.c)
        N = (det / (np.pi **3))** 0.5 * volume
        return N
    
    def grid_volume(self, number_of_points):
        # Create a grid of points in 3D space
        lengtha = np.linalg.norm(self.a)
        lengthb = np.linalg.norm(self.b)
        lengthc = np.linalg.norm(self.c)
        lengths = [lengtha, lengthb, lengthc]
        longest_item = max(lengths)    
        x = np.linspace(self.center[0]-longest_item, self.center[0]+ longest_item, number_of_points)
        y = np.linspace(self.center[1]-longest_item, self.center[1]+ longest_item, number_of_points)
        z = np.linspace(self.center[2]-longest_item, self.center[2]+ longest_item, number_of_points)
        points_in_ellipse = 0
        fake_points = 0
        matrixA = self.matrixA
        for i in x:
                for j in y:
                        for k in z:
                                point = np.array([[i - self.center[0]], [j - self.center[1]], [k - self.center[2]]])
                                transpose_r = np.transpose(point)
                                XTG = np.matmul(transpose_r, matrixA)
                                value = np.matmul(XTG, point) 
                                fake_match = False
                                value_match = False       
                                #if (i/lengtha)**2 + (j/lengthb)**2 + (k/lengthc)**2 < 1:
                                 #    fake_points += 1
                                  #   fake_match = True
                                if value[0] < 1.0:
                                    if math.isnan(value[0]) or math.isinf(value[0]):
                                        print("Warning: NaN or Inf value encountered in Gaussian grid volume calculation.")
                                    points_in_ellipse += 1
                                    value_match = True
                                #if not fake_match and value_match:
                                    print(f"Point ({i}, {j}, {k}) is inside the ellipse.")
                                #if fake_match and not value_match:
                                    print(f"Point ({i}, {j}, {k}) is outside the ellipse.")

        dx = 2 * longest_item / number_of_points
        dy = 2 * longest_item / number_of_points
        dz = 2 * longest_item / number_of_points
        point_volume = dx * dy * dz
        ellipse_volume = points_in_ellipse * point_volume
        return ellipse_volume
    
    def inside_ellipse(self, x, y, z):
        point = np.array([[x - self.center[0]], [y - self.center[1]], [z - self.center[2]]])
        transpose_r = np.transpose(point)
        XTG = np.matmul(transpose_r, self.matrixA)
        value = np.matmul(XTG, point) 
        if value[0] <= 1.0:
            return True
        else:
            return False
    

    def experiment_volume(self, number_of_points):
         #matrix A is the same 
        matrixA = self.matrixA
        inverse_A = inverse_of_matrix(matrixA)
        scale = 1.2
        max_x = (inverse_A[0][0] **0.5)
        max_y = (inverse_A[1][1] **0.5)
        max_z = (inverse_A[2][2] **0.5)
        number_of_points = int(number_of_points*scale)
        lengtha = np.linalg.norm(self.a)
        lengthb = np.linalg.norm(self.b)
        lengthc = np.linalg.norm(self.c)
        x = np.linspace(self.center[0]-max_x, self.center[0]+ max_x, number_of_points)
        y = np.linspace(self.center[1]-max_y, self.center[1]+ max_y, number_of_points)
        z = np.linspace(self.center[2]-max_z, self.center[2]+ max_z, number_of_points)
        points_in_ellipse = 0
        matrixA = self.matrixA
        for i in x:
                for j in y:
                        for k in z:
                                point = np.array([[i - self.center[0]], [j - self.center[1]], [k - self.center[2]]])
                                transpose_r = np.transpose(point)
                                XTG = np.matmul(transpose_r, matrixA)
                                value = np.matmul(XTG, point) 
                                value_match = False       
                                if value[0] <= 1.0 :
                                    if math.isnan(value[0]) or math.isinf(value[0]):
                                        print("Warning: NaN or Inf value encountered in Gaussian grid volume calculation.")
                                    points_in_ellipse += 1
                                    value_match = True


        dx = 2 * max_x / number_of_points
        dy = 2 * max_y / number_of_points
        dz = 2 * max_z / number_of_points
        point_volume = dx * dy * dz
        ellipse_volume = points_in_ellipse * point_volume
        return ellipse_volume
    
    
        

    def experiment_volume_fake_points(self, number_of_points):
         #matrix A is the same 
        matrixA = self.matrixA
        inverse_A = inverse_of_matrix(matrixA)
        scale = 1.2
        max_x = (inverse_A[0][0] **0.5)
        #removind scale of 1.2 and running again
        max_y = (inverse_A[1][1] **0.5)
        max_z = (inverse_A[2][2] **0.5)
        number_of_points = int(number_of_points*scale)
        lengtha = np.linalg.norm(self.a)
        lengthb = np.linalg.norm(self.b)
        lengthc = np.linalg.norm(self.c)
        x = np.linspace(self.center[0]-max_x, self.center[0]+ max_x, number_of_points)
        y = np.linspace(self.center[1]-max_y, self.center[1]+ max_y, number_of_points)
        z = np.linspace(self.center[2]-max_z, self.center[2]+ max_z, number_of_points)
        points_in_ellipse = 0
        fake_points = 0
        matrixA = self.matrixA
        for i in x:
                for j in y:
                        for k in z:
                                point = np.array([[i - self.center[0]], [j - self.center[1]], [k - self.center[2]]])
                                transpose_r = np.transpose(point)
                                XTG = np.matmul(transpose_r, matrixA)
                                value = np.matmul(XTG, point) 
                                fake_match = False
                                value_match = False       
                                if (i/lengtha)**2 + (j/lengthb)**2 + (k/lengthc)**2 <= 1:
                                     fake_points += 1
                                     fake_match = True
                                if value[0] <= 1.0 :
                                    if math.isnan(value[0]) or math.isinf(value[0]):
                                        print("Warning: NaN or Inf value encountered in Gaussian grid volume calculation.")
                                    points_in_ellipse += 1
                                    value_match = True
                                if not fake_match and value_match:
                                    print(f"Point ({i}, {j}, {k}) is inside the ellipse.")
                                if fake_match and not value_match:
                                    print(f"Point ({i}, {j}, {k}) is outside the ellipse.")

        dx = 2 * max_x / number_of_points
        dy = 2 * max_y / number_of_points
        dz = 2 * max_z / number_of_points
        point_volume = dx * dy * dz
        volume = points_in_ellipse * point_volume
        return ellipse_volume
    
    @classmethod
    def ellipse_intersection(cls, axis1, axis2, number_of_points):
        gaussianA = Gaussian.from_axes(axis1.a, axis1.b, axis1.c, axis1.center)
        gaussianB = Gaussian.from_axes(axis2.a, axis2.b, axis2.c, axis2.center)

        return cls.ellipse_intersection_volume(gaussianA, gaussianB, number_of_points)

    @classmethod
    def ellipse_intersection_volume(cls, gaussianA, gaussianB, number_of_points):
        matrixA = gaussianA.matrixA
        inverse_A = inverse_of_matrix(matrixA)
        #min is smallest axis
        max_xA = (inverse_A[0][0] **0.5) + gaussianA.center[0] 
        max_yA = (inverse_A[1][1] **0.5) + gaussianA.center[1]
        max_zA = (inverse_A[2][2] **0.5) + gaussianA.center[2]
        min_xA = gaussianA.center[0] - (inverse_A[0][0] **0.5) 
        min_yA = gaussianA.center[1] - (inverse_A[1][1] **0.5) 
        min_zA = gaussianA.center[2] - (inverse_A[2][2] **0.5) 
        matrixB = gaussianB.matrixA
        inverse_B = inverse_of_matrix(matrixB)
        max_xB = (inverse_B[0][0] **0.5) + gaussianB.center[0]
        max_yB = (inverse_B[1][1] **0.5) + gaussianB.center[1]
        max_zB = (inverse_B[2][2] **0.5) + gaussianB.center[2]
        min_xB = gaussianB.center[0] - (inverse_B[0][0] **0.5) 
        min_yB = gaussianB.center[1] - (inverse_B[1][1] **0.5) 
        min_zB = gaussianB.center[2] - (inverse_B[2][2] **0.5) 
        max_x = min(max_xA, max_xB)
        max_y = min(max_yA, max_yB)
        max_z = min(max_zA, max_zB)
        min_x = max(min_xA, min_xB)
        min_y = max(min_yA, min_yB)
        min_z = max(min_zA, min_zB)
        number_of_points = int(number_of_points)
        x = np.linspace(min_x, max_x, number_of_points)
        y = np.linspace(min_x, max_y, number_of_points)
        z = np.linspace(min_z, max_z, number_of_points)
        points_in_A = 0
        points_in_B = 0
        points_in_intersection = 0
        for i in x:
                for j in y:
                        for k in z:
                                point = np.array([[i], [j], [k]])
                                transpose_r = np.transpose(point)
                                XTGA = np.matmul(transpose_r, matrixA)
                                XTGB = np.matmul(transpose_r, matrixB)
                                valueA = np.matmul(XTGA, point) 
                                valueB = np.matmul(XTGB, point) 
                                if gaussianA.inside_ellipse(i, j, k) == True:
                                     points_in_A += 1
                                if gaussianA.inside_ellipse(i, j, k) == False:
                                    continue
                                if gaussianB.inside_ellipse(i, j, k) == True:
                                     points_in_B += 1
                                if gaussianA.inside_ellipse(i, j, k) == False:
                                     continue
                                if gaussianA.inside_ellipse(i, j, k) == True and gaussianB.inside_ellipse(i, j, k) == True:
                                    points_in_intersection += 1
        dx = (min_x - max_x) / number_of_points
        dy = (min_y - max_y) / number_of_points
        dz = (min_z - max_z) / number_of_points
        point_volume = dx * dy * dz
        ellipse_volume = points_in_intersection * point_volume
        return ellipse_volume

    @classmethod
    def gaussian_intersection(cls, gaussianA, gaussianB, number_of_points):
        matrixA = gaussianA.matrixA
        inverse_A = inverse_of_matrix(matrixA)
        matrixB = gaussianB.matrixA
        inverse_B = inverse_of_matrix(matrixB)
        C = 0.75225 * gaussianA.n ** (3/2) #w pi
        u = np.subtract(gaussianA.center, gaussianB.center)
        centerB = 0
        P = matrixA + matrixB
        PI = np.linalg.inv(P)
        PIA= np.matmul(PI, matrixA)
        v = np.matmul(PIA, u)
        vT = np.transpose(v)
        vTP = np.matmul(vT, P)
        vTpv = np.matmul(vTP, v)
        uT = np.transpose(u)
        uTA = np.matmul(uT, matrixA)
        uTAu = np.matmul(uTA, u)
        #not absolute value, magnitude
        volume = (( np.pi ** 3 / deteriminant_of_matrix(P))) ** 0.5 * (C ** 2) * np.exp(vTpv - uTAu)
        return volume

    @classmethod
    def plot_gaussian_intersection(cls, gaussianA, gaussianB, number_of_points):
        matrixA = gaussianA.matrixA
        inverse_A = inverse_of_matrix(matrixA)
        matrixB = gaussianB.matrixA
        inverse_B = inverse_of_matrix(matrixB)
        C = 0.75225 * gaussianA.n ** (3/2)#w pi
        u = np.subtract(gaussianA.center, gaussianB.center)
        centerB = 0
        P = matrixA + matrixB
        PI = np.linalg.inv(P)
        PIA= np.matmul(PI, matrixA)
        v = np.matmul(PIA, u)
        #might not be the most efficient, can revisitn
        U, D, VT = la.svd(P)
        axes_magnitudes = 1.0/np.sqrt(D)
        axes = VT * axes_magnitudes[:, np.newaxis]
        #gaussian rep for an ellipsoid 
        intersection_gaussian = Gaussian(P, v + gaussianB.center)
        gaussians = [gaussianA, gaussianB, intersection_gaussian]
        Gaussian.print_pymol_ellipse(gaussians, 'gaussian_intersection')
        
    #extract6 axis from P eith center v
    #don;t need to iterate thru all the points
    #convert to quadratric 
    #v offset by origional translate j l/
    #list of gaussions
    @classmethod
    def print_pymol_ellipse(cls, gaussians: List['Gaussian'], base: str) -> None:

        py_script = f'{base}.py'
        with open(py_script, 'wt') as fh:
            fh.write('from pymol.cgo import *\n')
            fh.write("cmd.delete('all')\n") 
            for ellipse_idx, gaussian in enumerate(gaussians):
                ellipse = quadratic_to_parametric(gaussian.center, gaussian.matrixA)
                center = ellipse.center
                mag = ellipse.axes_magnitudes
                rot = ellipse.eigen_vectors
                drawCommand = f'tmp{ellipse_idx} = drawEllipsoid([0.85, 0.85, 1.00] '
                for i in range(3):
                    drawCommand = drawCommand + f', {center[i]}'
                for i in range(3):
                    drawCommand = drawCommand + f', {mag[i]}'
                for i in range(3):
                    for j in range(3):
                        drawCommand = drawCommand + f', {rot[i][j]}'
                drawCommand = drawCommand + ')'
                fh.write(drawCommand)
                fh.write('\n')
                fh.write(f"cmd.load_cgo(tmp{ellipse_idx}, 'ellipsoid-cgo{ellipse_idx}')\n")
                fh.write(f"cmd.set('cgo_transparency', 0.5, 'ellipsoid-cgo{ellipse_idx}')\n")
                fh.write(f"obj{ellipse_idx} = [\n BEGIN, LINES, \n COLOR, 0, 1.0, 0, \n")
                # write axes
                for i in range(0,3):
                    fh.write(f'VERTEX, {center[0]}, {center[1]}, {center[2]},\n')
                    axis = ellipse.axes[i] + center
                    fh.write(f'VERTEX, {axis[0]}, {axis[1]}, {axis[2]},\n')
                fh.write("END\n] \n")
                fh.write(f"cmd.load_cgo(obj{ellipse_idx},'axis{ellipse_idx}')\n")
        full_py_path = py_script
        print(f'Pymol script {full_py_path}')
        #create gaussian output class  


        



                        
        
        
