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

# diabetic_patient = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]
# Pregnancies: 6
# Glucose: 148 (high)
# Blood Pressure: 72 (normal)
# Skin Thickness: 35 (elevated)
# Insulin: 0 (missing data — common in dataset)
# BMI: 33.6 (overweight/obese)
# Diabetes Pedigree Function: 0.627 (moderate family risk)
# Age: 50 (higher age is a risk factor)
# Patient likely to NOT have diabetes

# non_diabetic_patient = [[1, 85, 66, 29, 0, 26.6, 0.351, 31]]
# Pregnancies: 1
# Glucose: 85 (normal)
# Blood Pressure: 66 (normal)
# Skin Thickness: 29 (normal)
# Insulin: 0 (missing, but others are normal)
# BMI: 26.6 (healthy range)
# Diabetes Pedigree Function: 0.351 (low)
# Age: 31 (relatively young)

patient_name = str(input("What is Patient's Name?: "))
pregnancies = int(input(f'How many pregnancies has {patient_name} had?: '))
glucose = float(input(f'{patient_name} Plasma Glucose Concentration Level?: '))
blood_pressure = float(input(f'{patient_name} Blood Pressure(mm Hg)?: '))
skin_thickness = float(input(f'{patient_name} Triceps skin fold thickness(mm)?: '))
insulin = float(input(f'{patient_name} 2-Hour serum insulin(mu U/ml)?: '))
bmi = float(input(f'{patient_name} Body Mass Index(weight in kg/(height in m)^2)?: '))
diabetes_pedigree = float(input(f'{patient_name} Diabetes Pedigree Probability?: '))
age = int(input(f'{patient_name} Age(years)?: '))

patient = {'Name': patient_name,
           'Pregnancies': pregnancies,
           'Glucose': glucose,
           'Blood Pressure': blood_pressure,
           'Skin Thickness': skin_thickness,
           'Insulin': insulin,
           'BMI': bmi,
           'Diabetes Pedigree Function': diabetes_pedigree,
           'Age': age}

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
