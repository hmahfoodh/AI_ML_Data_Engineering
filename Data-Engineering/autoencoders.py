#autoencoders implementation to reconstruct data from a given dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

# Generate 1000 random data points in 2 dimensions
np.random.seed(42)
data = np.random.randn(1000, 2)
print(data)
print(len(data))

# Apply PCA for dimensionality reduction (optional, for comparison)
pca = PCA(n_components=1)
data_pca = pca.fit_transform(data)

# Create an autoencoder model
autoencoder = MLPRegressor(hidden_layer_sizes=(1,), activation='relu', solver='adam')

# Fit the autoencoder to the data
autoencoder.fit(data, data)

# Reconstruct the data using the autoencoder
data_reconstructed = autoencoder.predict(data)

# Plot the data before and after autoencoder
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c='blue', label='Original Data')
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.subplot(1, 2, 2)
plt.scatter(data_reconstructed[:, 0], data_reconstructed[:, 1], c='red', label='Reconstructed Data')
plt.title("Reconstructed Data")
plt.xlabel("Feature 1 (Reconstructed)")
plt.ylabel("Feature 2 (Reconstructed)")

plt.legend()
plt.tight_layout()
plt.show()

# When you reconstruct the data using the autoencoder, it is forced to project the 1-dimensional latent representation back into the original 2-dimensional space. This projection might not perfectly capture the original complexity of the data, resulting in a reduced number of distinct points in the reconstructed data.

# To obtain a more accurate reconstruction and potentially more diverse points, you can:

# Increase the latent dimension: Set n_components in the PCA to a higher value, allowing the autoencoder to learn a more complex representation.
# Use a deeper autoencoder: Add more hidden layers to the autoencoder to increase its capacity to capture more intricate patterns.
# Experiment with different activation functions: Different activation functions can impact the autoencoder's ability to represent complex relationships.
# By adjusting these parameters, you can explore how the reconstructed data changes and potentially obtain a more diverse set of points.