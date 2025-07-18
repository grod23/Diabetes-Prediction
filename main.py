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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import Model
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
import random

# Import Data

diabetes_data = pd.read_csv(r"C:\Users\gabe7\Downloads\diabetes.csv")


# print(diabetes_data.head())
# print(diabetes_data.shape)

# Data Shape: (768, 9)
# print(diabetes_data.info())


def main():
    torch.manual_seed(51)
    np.random.seed(51)
    random.seed(51)
    learning_rate = 0.005
    epochs = 1000
    position_weight = torch.tensor([500 / 268])
    weight_decay = 0.01

    model = Model()
    # ADAM Optimizer(Adding Weight Decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Binary Cross Entropy Loss(WithLogitsLoss combines sigmoid activation and BCE Loss in one function)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=position_weight)  # Added Weighted Loss

    # Set X and y Values

    # Train Test Split 80-20

    # stratify ensures class distribution is preserved in both training and test sets. Important for imbalanced
    # datasets like this one: (500 Non-Diabetic, 268 Diabetic)

    # First split: train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        diabetes_data.drop(columns="Outcome"),
        diabetes_data["Outcome"],
        test_size=0.2,
        stratify=diabetes_data["Outcome"],
        random_state=66
    )

    # Second split: train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.2,  # 20% of 80% = 16% → final split: 64% train, 16% val, 20% test
        stratify=y_temp,
        random_state=66
    )

    # Pre-Processing
    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert to Tensors

    # Shape should be (N, 1)
    # N = Number of Data Samples/Rows
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

    # print(X_train)
    # print(X_train.shape)

    # Train

    # Track Loss
    losses = []
    for epoch in range(epochs):
        model.train()
        y_hat = model(X_train)
        loss = loss_fn(y_hat, y_train)
        # Append Loss
        losses.append(loss.detach().numpy())
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = loss_fn(val_logits, y_val)

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

    # Loss VS Epoch
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.show()

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

    # After predictions
    print(classification_report(y_test.numpy(), preds.numpy(), digits=4))

    # Correct: 122/154
    # Accuracy: 0.7922
    #               precision    recall  f1-score   support
    #
    #          0.0     0.9595    0.7100    0.8161       100
    #          1.0     0.6375    0.9444    0.7612        54
    #
    #     accuracy                         0.7922       154
    #    macro avg     0.7985    0.8272    0.7886       154
    # weighted avg     0.8466    0.7922    0.7968       154
    # Dropout: 0.1
    # Weight_Decay .01
    # Epochs: 1000
    # LR: 0.005

    # Save and Reload Trained Model
    # Use nn.Sequential()

    # Hyper-Parameter Tuning:
    # Random Search


if __name__ == '__main__':
    main()
