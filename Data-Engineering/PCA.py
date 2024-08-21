import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate 5000 random data points in 3 dimensions
np.random.seed(42)
data = np.random.randn(5000, 3)

# Apply PCA with 2 components
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# Plot the data before and after PCA with different colors
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c='blue', label='Original Data')
plt.title("Original Data (3D)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.subplot(1, 2, 2)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c='green', label='PCA (2 components)')
plt.title("PCA (2 components)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.legend()
plt.tight_layout()
plt.show()