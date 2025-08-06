import matplotlib.pyplot as plt
file_path = 'gaussian_results.txt'
import numpy as np
#move function def up here
def read_gaussian_points(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    lines = content.strip().split('\n')
    points = []
    for line in lines:
        if line.strip():  # Check if the line is not empty
            values = line.split(',')
            if len(values) == 2:  # Ensure there are exactly two values
                try:
                    x, y = float(values[0].strip()), float(values[1].strip())
                    points.append((x, y))
                except ValueError:
                    continue  # Skip lines that cannot be converted to float
    return points
#call for x and y 
points = read_gaussian_points(file_path)
x = [point[0] for point in points]
y = [point[1] for point in points]
x_coord = np.linspace(min(x), max(x), num=100)
y_coord = x_coord
plt.figure(figsize=(10, 6))
plt.plot(x_coord, y_coord, label='y=x', color='red', linestyle='--')
plt.legend()
plt.scatter(x, y)
plt.xlabel('Expected Grid Value')
plt.ylabel('Gaussian Value')
plt.title('Expected Grid Value Versus Gaussian Value')
plt.grid(True)
plt.show()
covariance = np.cov(x, y)
x_standard_deviation = np.std(x)
y_standard_deviation = np.std(y)
cc = covariance/(x_standard_deviation * y_standard_deviation)
print(f"Correlation Coefficient: {cc}")