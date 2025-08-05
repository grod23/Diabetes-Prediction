# Prediction Model
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    # 8 Inputs, 2 Outputs
    def __init__(self, input_features=8, n1=32, n2=33, output_features=1, dropout_prob=0.1):
        super(Model, self).__init__()
        self.layer_1 = nn.Linear(input_features, n1)
        self.layer_2 = nn.Linear(n1, n2)
        self.output = nn.Linear(n2, output_features)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x_val):
        x_val = F.relu(self.layer_1(x_val))
        x_val = self.dropout(x_val)  # Dropout after ReLU
        x_val = F.relu(self.layer_2(x_val))
        x_val = self.dropout(x_val)  # Dropout after ReLU
        x_val = self.output(x_val)
        return x_val
