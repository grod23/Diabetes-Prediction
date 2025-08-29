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
import joblib

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
    learning_rate = 0.003
    epochs = 50
    position_weight = torch.tensor([500 / 268])  # Data is imbalanced: May decrease accuracy but will increase recall
    # on Class 1 which is most important
    weight_decay = 0.05
    batches = 64
    dropout_prob = .02

    model = Model(dropout_prob=dropout_prob)
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

    # Save Scaler
    joblib.dump(scaler, 'scaler.pkl')

    # Convert to Tensors

    # Shape should be (N, 1)
    # N = Number of Data Samples/Rows
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

    # Create Tensor Datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # Create DataLoaders

    train_loader = DataLoader(train_dataset, batch_size=batches, shuffle=True)  # Only want shuffle when training
    val_loader = DataLoader(val_dataset, batch_size=batches, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batches, shuffle=False)

    # Train

    # Track Loss
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            y_hat = model(batch_X)
            loss = loss_fn(y_hat, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Track average training loss for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_val, y_val in val_loader:
                val_logits = model(X_val)
                val_loss = loss_fn(val_logits, y_val)
                val_loss_total += val_loss.item()

                val_probs = torch.sigmoid(val_logits)
                val_preds = (val_probs > 0.5).float()
                val_correct += (val_preds == y_val).sum().item()
                val_total += y_val.size(0)

                # Recall
                true_positives = ((val_preds == 1) & (y_val == 1)).sum().item()
                actual_positives = (y_val == 1).sum().item()

        avg_val_loss = val_loss_total / len(val_loader)
        val_accuracy = val_correct / val_total
        recall = true_positives / actual_positives if actual_positives > 0 else 0

        # print(f"Epoch: {epoch}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, "
        # f"Val Accuracy: {val_accuracy:.4f}, Recall: {recall}")

    # Loss VS Epoch
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.show()

    # Testing
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_test, y_test in test_loader:
            logits = model(X_test)  # Raw outputs
            probs = torch.sigmoid(logits)  # Convert logits to probabilities
            preds = (probs > 0.5).float()  # Threshold to get binary predictions

            correct += (preds == y_test).sum().item()
            total += y_test.size(0)
            all_preds.append(preds)
            all_targets.append(y_test)

        accuracy = correct / total
        print(f'Correct: {correct}/{total}')
        print(f'Accuracy: {accuracy:.4f}')

    # Concatenate all predictions and targets
    all_preds_tensor = torch.cat(all_preds)
    all_targets_tensor = torch.cat(all_targets)
    # After predictions

    print(classification_report(all_targets_tensor.numpy(), all_preds_tensor.numpy(), digits=4))

    # Save and Reload Model
    torch.save(model.state_dict(), 'diabetes_model.pth')


if __name__ == '__main__':
    main()
