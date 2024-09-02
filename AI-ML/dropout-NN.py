import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from typing import OrderedDict


# Create a simple neural network model with dropout
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.2),  # 20% dropout rate
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Assuming you have training data (X_train, y_train) and validation data (X_val, y_val)
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))