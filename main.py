# About 2 in 7 adults are diagnosed with diabetes
# Datapoints:
#  Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, DiabetesPedigreeFunction, Age
# Outcome: 0 = No Diabetes. 1 = Diabetes
# From Data, 500 labeled as 0, 268 labeled as 1
# 8 Inputs
# 1 Output

# Binary Classification/Logistic Regression Problem
# Loss Function: BCE
# Outputs will be between 0 and 1. Don't want outputs less than 0
# Activation Function: ReLU
# Optimization Algorithm: ADAM

# Pre-Processing:
# Standardization(Data has a mean of 0 and Standard Deviation of 1)

# Train Test split starting at 80-20%

import torch
import torch.nn as nn
import torch.functional as F
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from model import Model

# Import Data

diabetes_data = pd.read_csv(r"C:\Users\gabe7\Downloads\diabetes.csv")


# print(diabetes_data.head())
# print(diabetes_data.shape)

# Data Shape: (768, 9)
# print(diabetes_data.info())


def main():
    torch.manual_seed(51)
    learning_rate = 0.01
    epochs = 10000

    model = Model()
    # ADAM Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Binary Cross Entropy Loss(WithLogitsLoss combines sigmoid activation and BCE Loss in one function)
    loss_fn = nn.BCEWithLogitsLoss()

    # Set X and y Values

    # Train Test Split 80-20

    # stratify ensures class distribution is preserved in both training and test sets. Important for imbalanced
    # datasets like this one: (500 Non-Diabetic, 268 Diabetic)
    X_train, X_test, y_train, y_test = \
        train_test_split(diabetes_data.loc[:, diabetes_data.columns != 'Outcome'], diabetes_data['Outcome'],
                         stratify=diabetes_data['Outcome'], random_state=66)

    # Pre-Processing
    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to Tensors

    # Shape should be (N, 1)
    # N = Number of Data Samples/Rows
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

    print(X_train)
    print(X_train.shape)

    # Train
    for epoch in range(epochs):
        y_hat = model(X_train)
        loss = loss_fn(y_hat, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Testing
    model.eval()
    with torch.no_grad():
        logits = model(X_test)  # Raw outputs
        probs = torch.sigmoid(logits)  # Convert logits to probabilities
        preds = (probs > 0.5).float()  # Threshold to get binary predictions

        correct = (preds == y_test).sum().item()
        total = y_test.size(0)
        accuracy = correct / total

        print(f'Correct: {correct}/{total}')
        print(f'Accuracy: {accuracy:.4f}')

        # Correct: 130/192
        # Accuracy: 67.71

        # Possible Issues:
        # No Regularization
        # No Early Stopping, Weight Decay, or Dropout
        # Model Architecture 8-->9-->10-->1
        # Accuracy may be misleading due to dataset imbalance

        # Possible Solutions:
        # Add Evaluation Metrics such as Precision, Recall, and F1 Score
        # Use a weighted BCE Loss
        # Use a Validation Accuracy during training to watch for overfitting
        # Use early stopping or fewer epochs

        # Plot Loss vs Epoch
        # Confusion Matrix
        # Save and Reload Trained Model
        # Use nn.Sequential()


if __name__ == '__main__':
    main()
