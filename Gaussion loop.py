from Gaussian import Gaussian

def test_random_ellipoid_generator():
            with open('gaussian_results.txt', 'a') as f:
                for i in range(100):
                    gaussianA, gaussianB = Gaussian.random_ellipsoid_generator_two()
                    #create a python file that runs things in a for loop and print out into a file
                    #run as a program not a test
                    #new file
                    #print out into a file, prints out x and y: grid volume and gaussian volume 
                    a1 = gaussianA.a
                    b1 = gaussianA.b
                    c1 = gaussianA.c
                    center1 = gaussianA.center
                    gaussian2 = Gaussian.from_axes(a1, b1, c1, center1)
                    a2 = gaussianB.a
                    b2 = gaussianB.b
                    c2 = gaussianB.c
                    center2 = gaussian2.center
                    number_of_points = 200
                    volume_gaussian = Gaussian.gaussian_intersection(gaussianA, gaussianB, number_of_points)                                                   
                    volume_ellipse = Gaussian.ellipse_intersection_volume(gaussianA, gaussianB, number_of_points)
                    f.write(f" {volume_gaussian}, {volume_ellipse}\n")
                    f.flush()
                    #write the results to a file

test_random_ellipoid_generator()