import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Generate sample data
np.random.seed(42)
x = np.random.randn(1000)
y = np.random.randn(1000)
time_series = np.random.randn(100)
df = pd.DataFrame({'x': x, 'y': y})

# Scatter plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 4, 1)
plt.scatter(x, y)
plt.title("Scatter Plot")

# Time series
plt.subplot(1, 4, 2)
plt.plot(time_series)
plt.title("Time Series")

# Histogram
plt.subplot(1, 4, 3)
plt.hist(x, bins=30)
plt.title("Histogram")

# Box plot
plt.subplot(1, 4, 4)
plt.boxplot(df['x'])
plt.title("Box Plot")

plt.tight_layout()
plt.show()