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

    def test_ellipse_volume(self):
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        c = np.array([0, 0, 1])
        volume = ellipse_volume(a, b, c)
        expected_volume = (4.0 / 3.0) * np.pi * (1) * (1) * (1)
        self.assertAlmostEqual(volume, expected_volume)
    
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
   
   
    def test_from_axes(self):
        a = [  0.03066212,   3.17282733,  -3.55495418]
        b = [  0.45286576,  -5.42733834,  -4.84004033]
        c = [-14.65331877,  -0.61805622,  -0.67800802]
        center = np.array([-0.23426652,  0.59141845,  0.64056746])
        result = Gaussian.from_axes(a, b, c, center)
        np.testing.assert_allclose(result.a, a)
        np.testing.assert_allclose(result.b, b)
        np.testing.assert_allclose(result.c, c)

    def test_volume_constant_smiles2round1(self):
        self.a = [ 0.8758055,   2.50142647,  1.24863099]
        self.b = [  0.33178312,  1.71381101,  -3.66606685]
        self.c = [-4.33687328,  1.39006208,  0.25733364]
        self.center = np.array([-6.94834066,  -0.32235233,  0.04795997])
        gaussian = Gaussian.from_axes(self.a, self.b, self.c, self.center)
        volume_const = gaussian.volume_constant()
        result = Gaussian.from_axes(self.a, self.b, self.c, self.center)
        np.testing.assert_allclose(result.a, self.a, rtol = 1e-4, atol = 1e-4)  
        np.testing.assert_allclose(result.b, self.b, rtol = 1e-4, atol = 1e-4)
        np.testing.assert_allclose(result.c, self.c, rtol = 1e-4, atol = 1e-4)
        return volume_const
    #volume constant is 0.7522527779826932

    def test_volume_constant_smiles2round2(self):
        self.a = [ -1.39014301,   -0.66195345,  1.65354358]
        self.b = [  1.59688158,  -2.63958842,  0.28581594]
        self.c = [2.97449481,  2.16407201,  3.36700415]
        self.center = np.array([-1.39665672,  -3.38629163,  -2.62219414])
        gaussian = Gaussian.from_axes(self.a, self.b, self.c, self.center)
        volume_const = gaussian.volume_constant()
        result = Gaussian.from_axes(self.a, self.b, self.c, self.center)
        np.testing.assert_allclose(result.a, self.a, rtol = 1e-4, atol = 1e-4)  
        np.testing.assert_allclose(result.b, self.b, rtol = 1e-4, atol = 1e-4)
        np.testing.assert_allclose(result.c, self.c, rtol = 1e-4, atol = 1e-4)
        return volume_const
    #volume constant is 0.7522527780636751

    def test_volume_constant_smiles1round1(self):
        self.a = [ -0.30003443,   1.32946329,  1.46748463]
        self.b = [  -2.79975352,  1.84930587,  -2.24779645]
        self.c = [-3.81354613,  -3.19881181,  2.11825574]
        self.center = np.array([-6.32351802,  0.27492053,  -0.93936663])
        gaussian = Gaussian.from_axes(self.a, self.b, self.c, self.center)
        volume_const = gaussian.volume_constant()
        result = Gaussian.from_axes(self.a, self.b, self.c, self.center)
        np.testing.assert_allclose(result.a, self.a, rtol = 1e-4, atol = 1e-4)  
        np.testing.assert_allclose(result.b, self.b, rtol = 1e-4, atol = 1e-4)
        np.testing.assert_allclose(result.c, self.c, rtol = 1e-4, atol = 1e-4)
        return volume_const
    #volume constant is 0.7522527780636752

    def test_volume_constant_smiles1round2(self):
        self.a = [ 0.49111521,   1.46922175,  -1.73698885]
        self.b = [  0.15652988,  -2.59142209,  -2.14768188]
        self.c = [-3.62847313,  0.37099802,  -0.71210589]
        self.center = np.array([0.49111521,  1.46922175,  -1.73698885])
        gaussian = Gaussian.from_axes(self.a, self.b, self.c, self.center)
        volume_const = gaussian.volume_constant()
        result = Gaussian.from_axes(self.a, self.b, self.c, self.center)
        np.testing.assert_allclose(result.a, self.a, rtol = 1e-4, atol = 1e-4)  
        np.testing.assert_allclose(result.b, self.b, rtol = 1e-4, atol = 1e-4)
        np.testing.assert_allclose(result.c, self.c, rtol = 1e-4, atol = 1e-4)
        return volume_const
    #volume constant is 0.752252778063675

    def test_volume_benzene(self):
        self.a = [ -0.01222658,   0.0466439,  1.93013602]
        self.b = [  -1.5133306,  2.51680727,  -0.07040777]
        self.c = [-2.69283929,  -1.61855984,  0.02205633]
        self.center = np.array([1.49194902e-04, 4.70517138e-05,  -1.71050511e-04])
        gaussian = Gaussian.from_axes(self.a, self.b, self.c, self.center)
        volume_const = gaussian.volume_constant()
        result = Gaussian.from_axes(self.a, self.b, self.c, self.center)
        np.testing.assert_allclose(result.a, self.a, rtol = 1e-4, atol = 1e-4)  
        np.testing.assert_allclose(result.b, self.b, rtol = 1e-4, atol = 1e-4)
        np.testing.assert_allclose(result.c, self.c, rtol = 1e-4, atol = 1e-4)
        return volume_const
    #volume constant is 0.7522527780636749

    def test_volume_ethane(self):
        self.a = [ -0.0891971,   -0.54384129,  1.69825358]
        self.b = [  0.00874386,  1.70106077,  0.54519951]
        self.c = [-2.45457096,  0.04891636,  -0.11325628]
        self.center = np.array([-2.29916708e-04, -1.92200344e-04,  -1.58862120e-05])
        gaussian = Gaussian.from_axes(self.a, self.b, self.c, self.center)
        volume_const = gaussian.volume_constant()
        result = Gaussian.from_axes(self.a, self.b, self.c, self.center)
        np.testing.assert_allclose(result.a, self.a, rtol = 1e-4, atol = 1e-4)  
        np.testing.assert_allclose(result.b, self.b, rtol = 1e-4, atol = 1e-4)
        np.testing.assert_allclose(result.c, self.c, rtol = 1e-4, atol = 1e-4)
        return volume_const
    
    #volume constant is 0.7522527780636752
    
    def test_row_matrix_multiplication(self):
        A = np.array([[-1, -1, -1]])
        B = np.array([[1,0,0], [0,1,0], [0,0,1]])
        expected_result = np.array([-1, -1, -1])
        result = row_matrix_multiplication(A, B)
        np.testing.assert_allclose(expected_result, result)
    
    def test_matrix_multiplication_column(self):
        A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        B = np.array([[-1], [-1], [-1]])
        expected_result = np.array([-1, -1, -1])
        result = matrix_multiplication_column(A, B)
        np.testing.assert_allclose(expected_result, result)

    def test_matrix_multiplication_row_column(self):
        A = np.array([[-1,-1,-1]])
        B = np.array([[-1], [-1], [-1]])
        expected_result = np.array([3])
        result = matrix_multiplication_row_column(A, B)
        np.testing.assert_allclose(expected_result, result)

    def test_grid_volume(self):
        a = [1, 0, 0]
        b = [0, 1, 0]
        c = [0, 0, 1]
        center = np.array([0, 0, 0])
        number_of_points = 100
        expected_volume = 4/3 * np.pi 
        gaussian = Gaussian.from_axes(a, b, c, center)
        volume = gaussian.grid_volume(number_of_points)
        #self.assertAlmostEqual(volume,expected_volume,0)
        relative_error = (expected_volume - volume) / expected_volume
        percentage_error = relative_error * 100
        self.assertLess(percentage_error, 5)
        #passes
 

    def test_grid_volume_smiles1_loop1(self):
        a = [-0.30003443, 1.32946329, 1.46748463]
        b = [-2.79975352, 1.84930587, -2.24779645]
        c = [-3.81354613, -3.19881181, 2.11825574]
        center = np.array([-6.32351802,  0.27492053,  -0.93936663])
        number_of_points = 100
        expected_volume = ellipse_volume(a, b, c)
        gaussian = Gaussian.from_axes(a, b, c, center)
        volume = gaussian.grid_volume(number_of_points)
        #self.assertAlmostEqual(volume,expected_volume,0)
        relative_error = (expected_volume - volume) / expected_volume
        percentage_error = relative_error * 100
        self.assertLess(percentage_error, 5)
        #expected volume is 136
        #grid volume is 183
        #NOW FAILS BY 5.4

 

    def test_grid_volume_benzene(self):
        a = [ -0.01222658,   0.0466439,  1.93013602]
        b = [  -1.5133306,  2.51680727,  -0.07040777]
        c = [-2.69283929,  -1.61855984,  0.02205633]
        center = np.array([1.49194902e-04, 4.70517138e-05,  -1.71050511e-04])
        number_of_points = 100
        expected_volume = ellipse_volume(a, b, c)
        gaussian = Gaussian.from_axes(a, b, c, center)
        volume = gaussian.grid_volume(number_of_points)
        #self.assertAlmostEqual(volume,expected_volume,0)
        relative_error = (expected_volume - volume) / expected_volume
        percentage_error = relative_error * 100
        self.assertLess(percentage_error, 5)
        #only fails by 1 even w increased grid


    def test_grid_volume_ethane(self):
        a = [ -0.0891971,   -0.54384129,  1.69825358]
        b = [  0.00874386,  1.70106077,  0.54519951]
        c = [-2.45457096,  0.04891636,  -0.11325628]
        self.center = np.array([-2.29916708e-04, -1.92200344e-04,  -1.58862120e-05])
        number_of_points = 100
        number_of_points = 100
        expected_volume = ellipse_volume(a, b, c)
        gaussian = Gaussian.from_axes(a, b, c, center)
        volume = gaussian.grid_volume(number_of_points)
        #self.assertAlmostEqual(volume,expected_volume,0)
        relative_error = (expected_volume - volume) / expected_volume
        percentage_error = relative_error * 100
        self.assertLess(percentage_error, 5)
        #fails by 0.97
    
    def test_smiles1_round2(self):
        a = [ 0.49111521,   1.46922175,  -1.73698885]
        b = [  0.15652988,  -2.59142209,  -2.14768188]
        c = [-3.62847313,  0.37099802,  -0.71210589]
        center = np.array([0.49111521,  1.46922175,  -1.73698885])
        number_of_points = 100
        gaussian = Gaussian.from_axes(a, b, c, center)
        expected_volume = ellipse_volume(a, b, c)
        gaussian = Gaussian.from_axes(a, b, c, center)
        volume = gaussian.grid_volume(number_of_points)
        #self.assertAlmostEqual(volume,expected_volume,0)
        relative_error = (expected_volume - volume) / expected_volume
        percentage_error = relative_error * 100
        self.assertLess(percentage_error, 5)
        #fails w 122 does not equal 173 at *2, now fails by 3.5
        #same assertation error at *1.1


    def test_grid_diagonal(self):
        a = [ 1,   0,  0]
        b = [ 0,  2,  0]
        c = [0,  0,  3]
        center = np.array([0,  0,  0])
        number_of_points = 100
        gaussian = Gaussian.from_axes(a, b, c, center)
        expected_volume = ellipse_volume(a, b, c)
        gaussian = Gaussian.from_axes(a, b, c, center)
        volume = gaussian.grid_volume(number_of_points)
        self.assertAlmostEqual(volume,expected_volume,0)
        relative_error = (expected_volume - volume) / expected_volume
        percentage_error = relative_error * 100
        self.assertLess(percentage_error, 5)
        #also fails with a large difference!
        #if 5 6 7 fails by 25 in 800s, same dif with 2 as 1.1, now fails by 15, 14.9 without scale
        # if 1 2 3 passes with a difference of 0.77139, about the same dif with 2 and 1.1, 0.78 without scale

    def test_experiment_volume(self):
        a = [-0.30003443, 1.32946329, 1.46748463]
        b = [-2.79975352, 1.84930587, -2.24779645]
        c = [-3.81354613, -3.19881181, 2.11825574]
        center = np.array([-6.32351802,  0.27492053,  -0.93936663])
        number_of_points = 100
        expected_volume = ellipse_volume(a, b, c)
        gaussian = Gaussian.from_axes(a, b, c, center)
        volume = gaussian.experiment_volume(number_of_points)
        #self.assertAlmostEqual(volume,expected_volume,0)
        relative_error = (expected_volume - volume) / expected_volume
        percentage_error = relative_error * 100
        self.assertLess(percentage_error, 5)
        #122 does not equal 183
        #is now 178 not 183 fails by 4.9, 4.59 without scale

    def test_experiment_volume_1(self):
        a = [ 4,   0,  0]
        b = [ 0,  5,  0]
        c = [0,  0,  6]
        center = np.array([0,  0,  0])
        number_of_points = 100
        gaussian = Gaussian.from_axes(a, b, c, center)
        expected_volume = ellipse_volume(a, b, c)
        gaussian = Gaussian.from_axes(a, b, c, center)
        volume = gaussian.experiment_volume(number_of_points)
        #self.assertAlmostEqual(volume,expected_volume,0)
        relative_error = (expected_volume - volume) / expected_volume
        percentage_error = relative_error * 100
        self.assertLess(percentage_error, 5)
        #fails by 15
        # fails by 12 in yoy8o the 500s, same without scale

    def test_experiment_smiles1_round2(self):
        a = [ 0.49111521,   1.46922175,  -1.73698885]
        b = [  0.15652988,  -2.59142209,  -2.14768188]
        c = [-3.62847313,  0.37099802,  -0.71210589]
        center = np.array([0.49111521,  1.46922175,  -1.73698885])
        number_of_points = 100
        gaussian = Gaussian.from_axes(a, b, c, center)
        expected_volume = ellipse_volume(a, b, c)
        gaussian = Gaussian.from_axes(a, b, c, center)
        volume = gaussian.experiment_volume(number_of_points)
        relative_error = (expected_volume - volume) / expected_volume
        percentage_error = relative_error * 100
        self.assertLess(percentage_error, 5)
        #self.assertAlmostEqual(volume,expected_volume,0)
        #fails by 3.055, comparied to other grid at 3.5, 3.04 without scale

    def test_experiment_smiles1_ethane(self):
        a = [ -0.0891971,   -0.54384129,  1.69825358]
        b = [  0.00874386,  1.70106077,  0.54519951]
        c = [-2.45457096,  0.04891636,  -0.11325628]
        center = np.array([-2.29916708e-04, -1.92200344e-04,  -1.58862120e-05])
        number_of_points = 100
        gaussian = Gaussian.from_axes(a, b, c, center)
        expected_volume = ellipse_volume(a, b, c)
        gaussian = Gaussian.from_axes(a, b, c, center)
        volume = gaussian.experiment_volume(number_of_points)
        #self.assertAlmostEqual(volume,expected_volume,0)
        relative_error = (expected_volume - volume) / expected_volume
        percentage_error = relative_error * 100
        self.assertLess(percentage_error, 5)
        #fails by 0.81, compared to 0.97, 0.82 without scale
        #volume of 32

    def test_ellipse_intersection_volume(self ):
        a1 = [ -0.0891971,   -0.54384129,  1.69825358]
        b1 = [  0.00874386,  1.70106077,  0.54519951]
        c1 = [-2.45457096,  0.04891636,  -0.11325628]
        center1 = np.array([-2.29916708e-04, -1.92200344e-04,  -1.58862120e-05])
        a2 = [ 0.49111521,   1.46922175,  -1.73698885]
        b2 = [  0.15652988,  -2.59142209,  -2.14768188]
        c2 = [-3.62847313,  0.37099802,  -0.71210589]
        center2 = np.array([0.49111521,  1.46922175,  -1.73698885])
        gaussianA = Gaussian.from_axes(a1, b1, c1, center1)
        gaussianB = Gaussian.from_axes(a2, b2, c2, center2)
        number_of_points = 100
        volume = Gaussian.ellipse_intersection_volume(gaussianA, gaussianB, number_of_points)
        #8641790

    def test_ellipse_intersection_volume_gaussian(self ):
        a1 = [ -0.0891971,   -0.54384129,  1.69825358]
        b1 = [  0.00874386,  1.70106077,  0.54519951]
        c1 = [-2.45457096,  0.04891636,  -0.11325628]
        center1 = np.array([-2.29916708e-04, -1.92200344e-04,  -1.58862120e-05])
        a2 = [ 0.49111521,   1.46922175,  -1.73698885]
        b2 = [  0.15652988,  -2.59142209,  -2.14768188]
        c2 = [-3.62847313,  0.37099802,  -0.71210589]
        center2 = np.array([0.49111521,  1.46922175,  -1.73698885])
        gaussianA = Gaussian.from_axes(a1, b1, c1, center1)
        gaussianB = Gaussian.from_axes(a2, b2, c2, center2)
        number_of_points = 100
        volume = Gaussian.gaussian_intersection(gaussianA, gaussianB, number_of_points)
        volume2 = volume * 2
        #actual gaussian for plot axis
        #two gaussians r the same

    def test_ellipse_intersection_volume_gaussian_same(self ):
        a1 = [ -0.0891971,   -0.54384129,  1.69825358]
        b1 = [  0.00874386,  1.70106077,  0.54519951]
        c1 = [-2.45457096,  0.04891636,  -0.11325628]
        center1 = np.array([-2.29916708e-04, -1.92200344e-04,  -1.58862120e-05])
        a2 = [ -0.0891971,   -0.54384129,  1.69825358]
        b2 = [  0.00874386,  1.70106077,  0.54519951]
        c2 = [-2.45457096,  0.04891636,  -0.11325628]
        center2 = np.array([-2.29916708e-04, -1.92200344e-04,  -1.58862120e-05])
        gaussianA = Gaussian.from_axes(a1, b1, c1, center1)
        gaussianB = Gaussian.from_axes(a2, b2, c2, center2)
        number_of_points = 100
        volume = Gaussian.gaussian_intersection(gaussianA, gaussianB, number_of_points)
        volume2 = volume * 2
        #8.73233
        
    def test_ellipse_intersection_volume_same(self ):
        a1 = [ -0.0891971,   -0.54384129,  1.69825358]
        b1 = [  0.00874386,  1.70106077,  0.54519951]
        c1 = [-2.45457096,  0.04891636,  -0.11325628]
        center1 = np.array([-2.29916708e-04, -1.92200344e-04,  -1.58862120e-05])
        a2 = [ -0.0891971,   -0.54384129,  1.69825358]
        b2 = [  0.00874386,  1.70106077,  0.54519951]
        c2 = [-2.45457096,  0.04891636,  -0.11325628]
        center2 = np.array([-2.29916708e-04, -1.92200344e-04,  -1.58862120e-05])
        gaussianA = Gaussian.from_axes(a1, b1, c1, center1)
        gaussianB = Gaussian.from_axes(a2, b2, c2, center2)
        number_of_points = 100
        volume = Gaussian.ellipse_intersection_volume(gaussianA, gaussianB, number_of_points)
        #26



















'''
    def test_volume_constant_smiles1(self):
        a = [  0.03066212,   3.17282733,  -3.55495418]
        b = [  0.45286576,  -5.42733834,  -4.84004033]
        c = [-14.65331877,  -0.61805622,  -0.67800802]
        center = np.array([-0.23426652,  0.59141845,  0.64056746])
        gaussian = Gaussian.from_axes(a, b, c, center)
        volume_const = gaussian.volume_constant(a, b, c, center)
        result = Gaussian.from_axes(a, b, c, center)
        np.testing.assert_allclose(result.a, a, rtol = 1e-5)
        np.testing.assert_allclose(result.b, b, rtol = 1e-5)
        np.testing.assert_allclose(result.c, c, rtol = 1e-5)
        return volume_const
    #volume constant is 0.752252778063675

    def test_volume_constant_benzene(self):
        a = [-6.30916155e-09, -6.54007375e-08, -3.75578650e-06]
        b = [-4.84691119e-01, -1.38956455e+00,  2.50111503e-02]
        c = [-1.79402064e+00,  6.25626765e-01, -7.88055598e-03]
        center = np.array([ 0.14019473,  0.02024879, -0.00058374])
        gaussian = Gaussian.from_axes(a, b, c, center)
        volume_const = gaussian.volume_constant(a, b, c, center)
        result = Gaussian.from_axes(a, b, c, center)
        np.testing.assert_allclose(result.a, a, rtol = 1e-5)
        np.testing.assert_allclose(result.b, b, rtol = 1e-5)
        np.testing.assert_allclose(result.c, c, rtol = 1e-5)
        return volume_const
        #volume constant is 0.7522979949497793

    def test_volume_constant_smiles2(self):
        a = [ 0.00469553, -0.00048851, -0.00077538]
        b = [-0.00048851,  0.02998697, -0.01254867]
        c = [-0.00077538, -0.01254867,  0.03283577]
        center = np.array([-0.23426652,  0.59141845,  0.64056746])
        gaussian = Gaussian.from_axes(a, b, c, center)
        volume_const = gaussian.volume_constant(a, b, c, center)
        result = Gaussian.from_axes(a, b, c, center)
        np.testing.assert_allclose(result.a, a, rtol = 1e-5)
        np.testing.assert_allclose(result.b, b, rtol = 1e-5)
        np.testing.assert_allclose(result.c, c, rtol = 1e-5)
        return volume_const
        #volume constant is 0.5294120241148772


    def test_volume_constant_smiles2_try1lloop2(self):
            a = [ -0.00469308, 0.0004882, 0.00078454]
            b = [0.00098383,  -0.01848606, 0.01769987]
            c = [0.01184756, 0.04220215,  0.04461031]
            center = np.array([-0.23426654,  0.59141845,  0.64056796])
            gaussian = Gaussian.from_axes(a, b, c, center)
            volume_const = gaussian.volume_constant(a, b, c, center)
            result = Gaussian.from_axes(a, b, c, center)
            np.testing.assert_allclose(result.a, a, rtol = 1e-2, atol = 1e-2)
            np.testing.assert_allclose(result.b, b, rtol = 1e-2, atol = 1e-2)
            np.testing.assert_allclose(result.c, c, rtol = 1e-2, atol = 1e-2)
            return volume_const
            #volume constant is 0.75218011136898
    
    def test_volume_constant_smiles2_try1lloop1(self):
            a = [ -0.00469308, 0.0004882, 0.00078454]
            b = [0.00098383,  -0.01898606, 0.01769987]
            c = [0.01184756, 0.04220215,  0.04461031]
            center = np.array([-0.23426654,  0.59141845,  0.64056796])
            gaussian = Gaussian.from_axes(a, b, c, center)
            volume_const = gaussian.volume_constant(a, b, c, center)
            result = Gaussian.from_axes(a, b, c, center)
            np.testing.assert_allclose(result.a, a, rtol = 1e-2, atol = 1e-2)
            np.testing.assert_allclose(result.b, b, rtol = 1e-2, atol = 1e-2)
            np.testing.assert_allclose(result.c, c, rtol = 1e-2, atol = 1e-2)
            return volume_const
            #volume constant is 0.75218011136898
            '''
