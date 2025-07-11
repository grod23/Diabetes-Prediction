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
