# This approach is useful if you have limited data for the target task. For example, if you have a small number of labeled reviews, you can use a pre-trained model to extract features from the reviews.
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the pre-trained model
pretrained_model = TfidfVectorizer()
pretrained_model.fit(['This is a positive review', 'This is a negative review'])

# Extract features from the data
X_train = pretrained_model.transform(['I love this product!', 'This product is terrible.'])

# Train a new model on the target task
y_train = np.array([1, 0])
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Make predictions on new data
X_new = pretrained_model.transform(['I love this product!'])
y_pred = clf.predict(X_new)

# Print the prediction
print(y_pred)