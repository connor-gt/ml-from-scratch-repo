import numpy as np
import matplotlib.pyplot as plt

from connor_ml.perceptron import Perceptron

inputs = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=float)

or_targets = np.array([0, 1, 1, 1], dtype=float)
and_targets = np.array([0, 0, 0, 1], dtype=float)
xor_targets = np.array([0, 1, 1, 0], dtype=float)
xnor_targets = np.array([1, 0, 0, 1], dtype=float)

# Change to any of the above (or, and, xor, xnor) to test logical functions
selected_targets = and_targets

perceptron = Perceptron(max_epochs=100, learning_rate=0.2, random_state=42)
perceptron.fit(inputs, selected_targets)

# The range of the x-axis
x1_range = np.linspace(-0.25, 1.25, 100) # 100 points from -0.25 to 1.25

# Apply the line equation: x2_boundary = -(w1 / w2) * x_range - (b / w2).
# See: https://www.geeksforgeeks.org/machine-learning/linear-separability-with-python/
w1 = perceptron.w_[0]
w2 = perceptron.w_[1]
b = perceptron.b_

# The perceptron's decision boundary is where the weighted sum equals 0, so this equation is: w1x1 + w2x2 + b = 0
# rearranged to solve for x2 (x1 becomes the x1_range, it's a collection of possible x1 values used to draw the
# line). Slope = -w1/w2, y-intercept = -b/w2.
# x2_boundary is a collection of values, and each element represents the x2 coordinate on the boundary for its
# corresponding x1 coordinate in x1_range.
x2_boundary = -(w1 / w2) * x1_range - (b / w2) 

# Plot the inputs
plt.scatter(inputs[:, 0], inputs[:, 1], c=selected_targets)
plt.xlabel("Input 1")
plt.ylabel("Input 2")

# Plot the learned boundary as a line between the data points
plt.plot(x1_range, x2_boundary) 
plt.show()
