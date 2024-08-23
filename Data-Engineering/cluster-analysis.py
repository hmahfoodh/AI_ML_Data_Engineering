import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(300, 2)

# Perform hierarchical clustering
model = AgglomerativeClustering(n_clusters=None, distance_threshold=0, linkage='ward')
model.fit(X)

# Calculate silhouette scores for different numbers of clusters
silhouette_scores = []
for k in range(2, 11):
    model = AgglomerativeClustering(n_clusters=k)
    model.fit(X)
    silhouette_scores.append(silhouette_score(X, model.labels_))

# Plot the elbow curve
plt.plot(range(2, 11), silhouette_scores)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Elbow Method")
plt.show()