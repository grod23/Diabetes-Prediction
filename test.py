import torch
import pandas as pd
import joblib
from model import Model

# Pregnancies: Number of times pregnant
# Glucose: Plasma glucose concentration 2 hours in an oral glucose tolerance test
# BloodPressure: Diastolic blood pressure (mm Hg)
# SkinThickness: Triceps skin fold thickness (mm)
# Insulin: 2-Hour serum insulin (mu U/ml)
# BMI: Body mass index (weight in kg/(height in m)^2)
# DiabetesPedigreeFunction: Likelihood of developing diabetes based off family history and genetics
# Age: Age (years)
# Outcome: Class variable (0 or 1)

model = Model()
model.load_state_dict(torch.load('diabetes_model.pth', weights_only=True))
scaler = joblib.load('scaler.pkl')
model.eval()

patient = {'Name': 'Diana',
           'Pregnancies': 2,
           'Glucose': 140,
           'Blood Pressure': 82,
           'Skin Thickness': 35,
           'Insulin': 150,
           'BMpip freeze > requirements.txt': 33.6,
           'Diabetes Pedigree Function': 0.627,
           'Age': 45}

input_data_frame = pd.DataFrame([[
    patient['Pregnancies'],
    patient['Glucose'],
    patient['Blood Pressure'],
    patient['Skin Thickness'],
    patient['Insulin'],
    patient['BMI'],
    patient['Diabetes Pedigree Function'],
    patient['Age']
]], columns=[
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age'
])

# Standardize Input
input_data = scaler.transform(input_data_frame)
# Convert to Pytorch Tensor
input_data = torch.tensor(input_data, dtype=torch.float32)

# Predictions
with torch.no_grad():
    output = model(input_data)
    probability = torch.sigmoid(output)
    prediction = (probability > 0.5).float()

print(f'Patient: {patient["Name"]} has a {probability.item()} chance of Diabetes')

