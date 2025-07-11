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
    model = Model()
    # ADAM Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    # Binary Cross Entropy Loss(WithLogitsLoss combines sigmoid activation and BCE Loss in one function)
    loss_fn = nn.BCEWithLogitsLoss()
    epochs = 500

    # Set X and y Values

    # Train Test Split 80-20

    # stratify ensures class distribution is preserved in both training and test sets. Important for imbalanced
    # datasets like this one: (500 Non-Diabetic, 268 Diabetic)
    X_train, X_test, y_train, y_test = \
        train_test_split(diabetes_data.loc[:, diabetes_data.columns != 'Outcome'], diabetes_data['Outcome'],
                         stratify=diabetes_data['Outcome'], random_state=66)


if __name__ == '__main__':
    main()
