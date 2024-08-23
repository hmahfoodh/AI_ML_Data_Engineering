import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = {'x': [1, 2, 3, 4, 5], 'y': [2, 4, 6, 8, 10]}
df = pd.DataFrame(data)

# Calculate correlation
correlation = df['x'].corr(df['y'])
print("Correlation coefficient:", correlation)

# Visualize correlation using a scatter plot
sns.scatterplot(x='x', y='y', data=df)
plt.title("Scatter Plot of x and y")
plt.show()