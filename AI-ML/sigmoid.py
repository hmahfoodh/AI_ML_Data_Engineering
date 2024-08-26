import numpy as np
import matplotlib.pyplot as plt

# Generate a range of x values
x = np.linspace(-5, 5, 100)

# Generate y values using a logistic function (similar to sigmoid)
y = 1 / (1 + np.exp(-(x - 2)))  # Adjust the offset (2) to shift the curve

# Add some noise to the data
y += np.random.normal(0, 0.1, size=100)  # Adjust the standard deviation for noise level

# Plot the data and the sigmoid function
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label="Generated Data")

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Plot the sigmoid function
y_sigmoid = sigmoid(x)
plt.plot(x, y_sigmoid, label="Sigmoid Function")

plt.title("Sigmoid Function and Generated Data")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()