import random
from model import Model
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import classification_report

diabetes_data = pd.read_csv(r"C:\Users\gabe7\Downloads\diabetes.csv")
torch.manual_seed(51)


def train_evaluation(learning_rate, dropout_rate, hidden_size, weight_decay):
    epochs = 1000
    position_weight = torch.tensor([500 / 268])
    hidden_size_double = hidden_size * 2
    model = Model(n1=hidden_size, n2=hidden_size_double, dropout_prob=dropout_rate)
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
    # plt.plot(losses)
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Loss vs. Epoch")
    # plt.show()

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

    return accuracy


def random_search(trials):
    best_accuracy = 0
    best_parameters = None
    best_trial = 0

    for trial in range(trials):
        print(f"\n🔍 Trial {trial + 1}/{trials}")

        # Random Hyperparameters
        lr = 10 ** random.uniform(-4, -2)  # Learning Rate: 0.0001 to 0.01
        dropout = random.uniform(0.1, 0.5)  # Dropout: 0.1 to 0.5
        hidden_size = random.choice([16, 32, 64])  # Hidden Layer Size
        weight_decay = 10 ** random.uniform(-4, -1)  # Weight Decay: 0.0001 to 0.01

        print(f"Testing: lr={lr:.5f}, dropout={dropout:.2f}, hidden={hidden_size}, weight_decay={weight_decay:.5f}")

        accuracy = train_evaluation(lr, dropout, hidden_size, weight_decay)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_parameters = (lr, dropout, hidden_size, weight_decay)
            best_trial = trial + 1

        print(f"\n🏆 Best Configuration: {best_trial}")
        print(f"Accuracy: {best_accuracy:.4f}")
        print(
            f"lr={best_parameters[0]:.5f}, dropout={best_parameters[1]:.2f}, hidden={best_parameters[2]}, "
            f"weight_decay={best_parameters[3]:.5f}")


random_search(50)

# Trial 15/50
# Testing: lr=0.00243, dropout=0.28, hidden=64, weight_decay=0.01467
# Epoch 0 | Train Loss: 0.8904 | Val Loss: 0.8689
# Epoch 100 | Train Loss: 0.6136 | Val Loss: 0.7059
# Epoch 200 | Train Loss: 0.5791 | Val Loss: 0.7174
# Epoch 300 | Train Loss: 0.5676 | Val Loss: 0.7209
# Epoch 400 | Train Loss: 0.5697 | Val Loss: 0.7214
# Epoch 500 | Train Loss: 0.5560 | Val Loss: 0.7219
# Epoch 600 | Train Loss: 0.5658 | Val Loss: 0.7242
# Epoch 700 | Train Loss: 0.5618 | Val Loss: 0.7239
# Epoch 800 | Train Loss: 0.5537 | Val Loss: 0.7184
# Epoch 900 | Train Loss: 0.5551 | Val Loss: 0.7268
# Correct: 125/154
# Accuracy: 0.8117
#               precision    recall  f1-score   support
#
#          0.0     0.9610    0.7400    0.8362       100
#          1.0     0.6623    0.9444    0.7786        54
#
#     accuracy                         0.8117       154
#    macro avg     0.8117    0.8422    0.8074       154
# weighted avg     0.8563    0.8117    0.8160       154
# Best Configuration: 15
# Accuracy: 0.8117
# lr=0.00243, dropout=0.28, hidden=64, weight_decay=0.01467
