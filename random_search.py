import random
from model import Model
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import optuna

diabetes_data = pd.read_csv(r"C:\Users\gabe7\Downloads\diabetes.csv")
torch.manual_seed(51)


# Hyper Parameters to optimize:
# Learning Rate
# Epochs and Batch Size
# Weight Decay Rate
# Dropout Probability
# Validation/Train/Test Split
# Hidden Layer Neuron Size

def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    epochs = trial.suggest_categorical("epochs", [10, 50, 100, 250, 500, 1000])
    batches = trial.suggest_categorical("batches", [10, 16, 32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    # hidden_size = trial.suggest_int("hidden_size", 32, 256)

    # Call train() Function
    torch.manual_seed(51)
    np.random.seed(51)
    random.seed(51)
    position_weight = torch.tensor([500 / 268])

    model = Model(dropout_prob=dropout)
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

    # Create Tensor Datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Create DataLoaders

    train_loader = DataLoader(train_dataset, batch_size=batches, shuffle=True)  # Only want shuffle when training
    val_loader = DataLoader(val_dataset, batch_size=batches, shuffle=False)

    # Train

    # Track Loss
    losses = []

    for epoch in range(epochs):
        model.train()

        for batch_X, batch_y in train_loader:
            y_hat = model(batch_X)
            loss = loss_fn(y_hat, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_val, y_val in val_loader:
                val_logits = model(X_val)
                val_probs = torch.sigmoid(val_logits)
                val_preds = (val_probs > 0.5).float()
                val_correct += (val_preds == y_val).sum().item()
                val_total += y_val.size(0)

    val_accuracy = val_correct / val_total
    return val_accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)
print("Best hyperparameters:", study.best_params)
print(f"Best validation accuracy: {study.best_value:.4f}")

# learNing_rate': 0.0012526868639613671, 'epochs': 50, 'batches': 64, 'dropout': 0.10229476735256574,
# 'weight_decay': 1.037538434833678e-06, 'hidden_size': 132}

# Best hyperparameters: {'learning_rate': 0.003067127556949599, 'epochs': 50, 'batches': 16,
# 'dropout': 0.13888900962321554, 'weight_decay': 3.9538410247089555e-05, 'hidden_size': 50}
# Best validation accuracy: 0.7967
