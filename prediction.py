import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import joblib

def create_prediction_model(data_path):
    
    df = pd.read_csv(data_path)
    
    columns = df.columns.tolist()
    columns.remove('LUNG_CANCER')  
    
    with open('feature_columns.txt', 'w') as f:
        f.write(','.join(columns))
    
    le = LabelEncoder()
    df['GENDER'] = le.fit_transform(df['GENDER'])
    df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])
    
    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_scaled, y)
    
    joblib.dump(rf_model, 'lung_cancer_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(le, 'label_encoder.joblib')
    
    return rf_model, scaler, X.columns

def get_feature_names():
    
    with open('feature_columns.txt', 'r') as f:
        return f.read().split(',')

def predict_lung_cancer():
    
    model = joblib.load('lung_cancer_model.joblib')
    scaler = joblib.load('scaler.joblib')
    
    feature_names = get_feature_names()
    
    print("\nPlease enter patient information:")
    patient_data = {}
    
    for feature in feature_names:
        if feature == 'GENDER':
            value = input(f"{feature} (M/F): ")
            patient_data[feature] = 1 if value.upper() == 'M' else 0
        elif feature == 'AGE':
            patient_data[feature] = int(input(f"{feature}: "))
        else:
            patient_data[feature] = int(input(f"{feature} (0/1): "))
    
    patient_df = pd.DataFrame([patient_data])[feature_names]
    
    patient_scaled = scaler.transform(patient_df)
    
    prediction = model.predict(patient_scaled)
    prediction_proba = model.predict_proba(patient_scaled)
    
    return prediction[0], prediction_proba[0], patient_data

def visualize_prediction(prediction, prediction_proba, patient_data, model):
    
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 2, 1)
    probabilities = prediction_proba
    labels = ['No Cancer', 'Cancer']
    colors = ['green', 'red']
    bars = plt.bar(labels, probabilities, color=colors)
    plt.title('Cancer Prediction Probability', fontsize=12, pad=20)
    plt.ylabel('Probability')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom')
    
    plt.subplot(2, 2, 2)
    feature_importance = pd.DataFrame({
        'feature': patient_data.keys(),
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title('Feature Importance in Prediction', fontsize=12, pad=20)
    plt.xlabel('Importance Score')
    
    plt.subplot(2, 2, 3)
    
    risk_factors = {k: v for k, v in patient_data.items() 
                   if k not in ['AGE', 'GENDER']}
    
    factors = list(risk_factors.keys())
    values = list(risk_factors.values())
    
    colors = ['red' if v == 1 else 'green' for v in values]
    
    bars = plt.barh(factors, values, color=colors)
    
    for i, v in enumerate(values):
        label = 'Present' if v == 1 else 'Absent'
        plt.text(v, i, f'  {label}', va='center')
    
    plt.title('Patient Risk Factors Status', fontsize=12, pad=20)
    plt.xlabel('Status (0: Absent, 1: Present)')
    
    demo_text = f"Patient Demographics:\nAge: {patient_data['AGE']}\nGender: {'Male' if patient_data['GENDER'] == 1 else 'Female'}"
    plt.text(1.2, len(factors)/2, demo_text, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.subplot(2, 2, 4)
    df = pd.read_csv('lung_cancer_data.csv')
    sns.histplot(data=df, x='AGE', bins=20, color='skyblue')
    plt.axvline(x=patient_data['AGE'], color='red', linestyle='--', label='Patient Age')
    plt.title('Age Distribution with Patient Age', fontsize=12, pad=20)
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    plt.suptitle('Lung Cancer Risk Assessment Dashboard', fontsize=14, y=0.95)
    
    plt.show()

def main():
    print("Creating prediction model...")
    model, scaler, features = create_prediction_model('lung_cancer_data.csv')
    
    while True:
        prediction, prediction_proba, patient_data = predict_lung_cancer()
        
        print("\nPrediction Results:")
        print("-" * 50)
        print(f"Cancer Risk: {'High' if prediction == 1 else 'Low'}")
        print(f"Probability of Cancer: {prediction_proba[1]:.2%}")
        print(f"Probability of No Cancer: {prediction_proba[0]:.2%}")
        
        visualize_prediction(prediction, prediction_proba, patient_data, model)
        
        if input("\nWould you like to make another prediction? (y/n): ").lower() != 'y':
            break

if __name__ == "__main__":
    main()