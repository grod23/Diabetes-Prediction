# Diabetes-Prediction
About 2 in 7 adults are diagnosed with diabetes affecting millions worldwide. Diabetes disrupts how the body regulatutes sugar, often leading to serious helath conditions such as heart disease, kidney failure, and nerve damage. This project leverages a predictive model that utilizes 8 key health indicators to assess a patients likeilhood of having diabetes. The models anaylsis of these inputs aims to support early diagonisis and to assist health care providers. 
This model takes the following 8 inputs: 
1. **Pregnancies** — Number of times the patient has been pregnant
2. **Glucose** — Plasma glucose concentration (mg/dL)
3. **Blood Pressure** — Diastolic blood pressure (mm Hg)
4. **Skin Thickness** — Triceps skin fold thickness (mm)
5. **Insulin** — 2-Hour serum insulin (mu U/ml)
6. **BMI** — Body Mass Index (weight in kg/(height in m)^2)
7. **Diabetes Pedigree Function** — A function that scores likelihood of diabetes based on family history
8. **Age** — Age of the patient (years)
## Installation
**Clone the repository**
```bash
git clone https://github.com/grod23/Diabetes-Prediction.git
cd Diabetes-Prediction1
```

### Create Virtual enviroment
```python -m venv venv
source venv/bin/activate       # macOS/Linux
or
venv\Scripts\activate          # Windows
```
### Install Required Packages
```
pip install -r requirements.txt
```
### Run Script
```
python test.py
```
# Classification Report

```
                precision    recall  f1-score   support
          0.0     0.9487    0.7400    0.8315       100
          1.0     0.6579    0.9259    0.7692        54

     accuracy                         0.8052       154
    macro avg     0.8033    0.8330    0.8003       154
 weighted avg     0.8467    0.8052    0.8096       154 
```
# Conclusion

The model favors a **high recall** over precision for diabetic cases, which is ideal in a medical screening context. In these scenarios, **false negatives (missing a true diabetic)** are far more dangerous than **false positives (flagging a non-diabetic as diabetic)**. Model exhibits an **80.52%** accuracy and a strong performance in identifying potential diabetic patients. With a high recall of **92.59%**, the model ensures that most diabetic cases are flagged for further medical attention. While the precision may be lower, this is an acceptable trade-off for an early detection system specifically designed to catch as many diabetic cases as possible for a follow-up screening. All in all, there is defintely room for improvement for larger and more diverse datasets, but the model provides a promising foundation for data-centric tools to be used in medical diagnostics. 

