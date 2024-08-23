import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Generate 5000 random data points
np.random.seed(42)  # Set a seed for reproducibility
data = np.random.randn(5000, 3)  # 5000 points, 3 features

# Create a DataFrame
df = pd.DataFrame(data, columns=['Feature 1', 'Feature 2', 'Feature 3'])

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Create the heatmap with adjusted parameters
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='viridis', annot_kws={'fontsize': 12})

# Set the title
plt.title('Correlation Heatmap')

# Show the plot
plt.show()