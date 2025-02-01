import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
from datetime import datetime

class LungCancerReport:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.model = None
        self.scaler = None
        self.feature_importance = None
        
    def generate_dataset_summary(self):
        """Generate summary statistics of the dataset"""
        summary = {
            'total_samples': len(self.df),
            'positive_cases': len(self.df[self.df['LUNG_CANCER'] == 'YES']),
            'negative_cases': len(self.df[self.df['LUNG_CANCER'] == 'NO']),
            'features': len(self.df.columns) - 1,  
            'gender_distribution': self.df['GENDER'].value_counts().to_dict(),
            'age_stats': self.df['AGE'].describe().to_dict()
        }
        return summary

    def analyze_risk_factors(self):
        """Analyze the prevalence and correlation of risk factors"""
        risk_factors = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
                       'CHRONIC DISEASE', 'WHEEZING', 'ALCOHOL CONSUMING', 
                       'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 
                       'CHEST PAIN']
        
        risk_analysis = {
            'prevalence': self.df[risk_factors].mean().sort_values(ascending=False).to_dict(),
            'correlation_matrix': self.df[risk_factors].corr().round(2).to_dict()
        }
        return risk_analysis

    def train_model(self):
        """Train and evaluate the prediction model"""
        le = LabelEncoder()
        df_processed = self.df.copy()
        df_processed['GENDER'] = le.fit_transform(df_processed['GENDER'])
        df_processed['LUNG_CANCER'] = le.fit_transform(df_processed['LUNG_CANCER'])
        
        X = df_processed.drop('LUNG_CANCER', axis=1)
        y = df_processed['LUNG_CANCER']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced', None]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        self.scaler = scaler
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'best_parameters': grid_search.best_params_,
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }
        
        self.feature_importance = metrics['feature_importance']
        return metrics

    def generate_visualizations(self, output_path='report_visualizations'):
        
        import os
        os.makedirs(output_path, exist_ok=True)
        
        sns.set_style("whitegrid")
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x='AGE', hue='LUNG_CANCER', multiple="stack")
        plt.title('Age Distribution by Lung Cancer Status')
        plt.savefig(f'{output_path}/age_distribution.png')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        risk_factors = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
                       'CHRONIC DISEASE', 'WHEEZING', 'ALCOHOL CONSUMING', 
                       'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 
                       'CHEST PAIN']
        sns.heatmap(self.df[risk_factors].corr(), annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Risk Factors Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'{output_path}/correlation_matrix.png')
        plt.close()
        
        if self.feature_importance:
            plt.figure(figsize=(10, 6))
            importance_df = pd.DataFrame({
                'Feature': self.feature_importance.keys(),
                'Importance': self.feature_importance.values()
            }).sort_values('Importance', ascending=False)
            
            sns.barplot(data=importance_df, x='Importance', y='Feature')
            plt.title('Feature Importance in Prediction Model')
            plt.tight_layout()
            plt.savefig(f'{output_path}/feature_importance.png')
            plt.close()
        
        return output_path

    def generate_report(self, output_file='lung_cancer_analysis_report.md'):
        """Generate a comprehensive markdown report"""
        # Get all analyses
        summary = self.generate_dataset_summary()
        risk_analysis = self.analyze_risk_factors()
        model_metrics = self.train_model()
        viz_path = self.generate_visualizations()
        
        # Create report
        report = f"""# Lung Cancer Analysis and Prediction Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Dataset Overview
- Total number of samples: {summary['total_samples']}
- Positive cases: {summary['positive_cases']}
- Negative cases: {summary['negative_cases']}
- Number of features: {summary['features']}

### Gender Distribution
- Male: {summary['gender_distribution']['M']}
- Female: {summary['gender_distribution']['F']}

### Age Statistics
- Mean age: {summary['age_stats']['mean']:.2f}
- Minimum age: {summary['age_stats']['min']}
- Maximum age: {summary['age_stats']['max']}
- Standard deviation: {summary['age_stats']['std']:.2f}

## 2. Risk Factor Analysis
### Top 5 Most Prevalent Risk Factors
{format_risk_factors(list(risk_analysis['prevalence'].items())[:5])}

## 3. Model Performance
### Model Metrics
- Accuracy: {model_metrics['accuracy']:.3f}
- Precision: {model_metrics['precision']:.3f}
- Recall: {model_metrics['recall']:.3f}
- F1 Score: {model_metrics['f1_score']:.3f}

### Best Model Parameters
{format_parameters(model_metrics['best_parameters'])}

### Top 5 Most Important Features
{format_feature_importance(list(model_metrics['feature_importance'].items())[:5])}

## 4. Visualizations
The following visualizations have been generated and saved in the '{viz_path}' directory:
1. Age Distribution by Lung Cancer Status
2. Risk Factors Correlation Matrix
3. Feature Importance Plot

## 5. Conclusions and Recommendations
1. Based on the analysis, the most significant risk factors for lung cancer prediction are:
   {format_top_features(list(model_metrics['feature_importance'].items())[:3])}

2. The model shows {interpret_model_performance(model_metrics['f1_score'])} performance with an F1 score of {model_metrics['f1_score']:.3f}.

3. Key findings:
   - {generate_key_findings(risk_analysis, model_metrics)}

## 6. Model Usage Instructions
The trained model has been saved and can be used for making predictions using the provided prediction interface. For new predictions:
1. Use the standardized input format
2. Consider all risk factors
3. Ensure all required features are provided

Note: This model should be used as a screening tool only and not as a definitive diagnostic tool.
"""
        
        with open(output_file, 'w') as f:
            f.write(report)
            
        return output_file

def format_risk_factors(risk_factors):
    return '\n'.join([f"- {factor}: {value*100:.1f}%" for factor, value in risk_factors])

def format_parameters(parameters):
    return '\n'.join([f"- {param}: {value}" for param, value in parameters.items()])

def format_feature_importance(features):
    return '\n'.join([f"- {feature}: {importance:.3f}" for feature, importance in features])

def format_top_features(features):
    return ', '.join([f"{feature} ({importance:.3f})" for feature, importance in features])

def interpret_model_performance(f1_score):
    if f1_score > 0.9:
        return "excellent"
    elif f1_score > 0.8:
        return "good"
    elif f1_score > 0.7:
        return "fair"
    else:
        return "moderate"

def generate_key_findings(risk_analysis, model_metrics):
    top_risk = max(risk_analysis['prevalence'].items(), key=lambda x: x[1])
    return f"The most common risk factor is {top_risk[0]} with a prevalence of {top_risk[1]*100:.1f}%"

def main():
    report_generator = LungCancerReport('lung_cancer_data.csv')
    
    report_file = report_generator.generate_report()
    
    print(f"Report generated successfully: {report_file}")
    
    joblib.dump(report_generator.model, 'lung_cancer_model.joblib')
    joblib.dump(report_generator.scaler, 'scaler.joblib')

if __name__ == "__main__":
    main()