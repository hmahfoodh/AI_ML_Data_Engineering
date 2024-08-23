import pandas as pd
import numpy as np
from scipy.stats import mode

# Sample data
data = [1, 2, 3, 4, 5, 5, 5, 6, 7, 8]

# Calculate summary statistics
mean = np.mean(data)
median = np.median(data)
mode = mode(data)
std_dev = np.std(data)
variance = np.var(data)
skewness = pd.Series(data).skew()
kurtosis = pd.Series(data).kurtosis()

print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
print("Standard Deviation:", std_dev)
print("Variance:", variance)
print("Skewness:", skewness)
print("Kurtosis:", kurtosis)