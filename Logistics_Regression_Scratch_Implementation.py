import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

class LogisticRegressionScratch:
    def __init__(self, lr=0.01, num_iter=1000):
        self.lr = lr
        self.num_iter = num_iter
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return np.where(y_predicted >= 0.5, 1, 0)

# Load dataset
data = pd.read_csv("dataset/heart_data_set.csv")
X = data.drop("target", axis=1).values
y = data["target"].values

# Normalize data
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegressionScratch(lr=0.01, num_iter=10000)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy (Logistic Regression from Scratch): {accuracy:.4f}")

# Save model
os.makedirs("model_from_scratch", exist_ok=True)
joblib.dump(model, "model_from_scratch/logistic_regression_model.pkl")
