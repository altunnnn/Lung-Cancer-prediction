import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_and_explore_data(filepath):
    
    df = pd.read_csv(filepath)
    
    print("\nDataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    
    print("\nNumerical Features Summary:")
    print(df.describe())
    
    print("\nGender Distribution:\n", df['GENDER'].value_counts(normalize=True))
    print("\nSmoking Distribution:\n", df['SMOKING'].value_counts(normalize=True))
    print("\nLung Cancer Distribution:\n", df['LUNG_CANCER'].value_counts(normalize=True))
    
    return df

def visualize_data_distribution(df):
  
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='AGE', hue='LUNG_CANCER', multiple="stack", bins=20)
    plt.title('Age Distribution by Lung Cancer Status')
    
    plt.subplot(2, 2, 2)
    sns.countplot(data=df, x='GENDER', hue='LUNG_CANCER')
    plt.title('Gender Distribution by Lung Cancer Status')
    
    plt.subplot(2, 2, 3)
    sns.countplot(data=df, x='SMOKING', hue='LUNG_CANCER')
    plt.title('Smoking Distribution by Lung Cancer Status')
    
    plt.subplot(2, 2, 4)
    symptoms = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
               'CHRONIC DISEASE', 'WHEEZING', 'ALCOHOL CONSUMING', 
               'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 
               'CHEST PAIN']
    correlation = df[symptoms].corr()
    sns.heatmap(correlation, cmap='coolwarm', square=True, annot=True, fmt='.2f')
    plt.title('Correlation between Symptoms')
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    symptom_means = df[symptoms].mean().sort_values(ascending=True)
    sns.barplot(x=symptom_means.values, y=symptom_means.index)
    plt.title('Prevalence of Each Symptom')
    plt.xlabel('Proportion of Patients')
    plt.tight_layout()
    plt.show()

def preprocess_data(df):
    """
    Clean and preprocess the dataset
    """
    df_processed = df.copy()
    
    if df_processed.isnull().sum().any():
        print("\nHandling missing values...")
        df_processed = df_processed.fillna(df_processed.median())
    
    le = LabelEncoder()
    df_processed['GENDER'] = le.fit_transform(df_processed['GENDER'])
    df_processed['LUNG_CANCER'] = le.fit_transform(df_processed['LUNG_CANCER'])
    
    X = df_processed.drop('LUNG_CANCER', axis=1)
    y = df_processed['LUNG_CANCER']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y

def analyze_feature_importance(X, y):
    """
    Analyze and visualize feature importance using multiple methods
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance, x='Importance', y='Feature')
    plt.title('Feature Importance for Lung Cancer Prediction')
    plt.tight_layout()
    plt.show()
    
    print("\nFeature Importance Analysis:")
    print("-" * 50)
    print(feature_importance.to_string(index=False))
    
    return feature_importance

def train_and_evaluate_models(X, y):
    """
    Train and evaluate multiple models with cross-validation and hyperparameter tuning
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced', None]
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy'],
                'class_weight': ['balanced', None]
            }
        },
        'SVM': {
            'model': SVC(random_state=42),
            'params': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'class_weight': ['balanced', None]
            }
        }
    }
    
    results = {}
    confusion_matrices = {}
    best_models = {}
    cv_scores = {}
    
    for name, model_info in models.items():
        print(f"\nTraining {name}...")
        
        grid_search = GridSearchCV(
            model_info['model'],
            model_info['params'],
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_models[name] = grid_search.best_estimator_
        
        y_pred = grid_search.predict(X_test)
        
        cv_scores[name] = grid_search.cv_results_['mean_test_score']
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'best_params': grid_search.best_params_,
            'cv_mean_score': grid_search.best_score_,
            'cv_std': grid_search.cv_results_['std_test_score'][grid_search.best_index_]
        }
        
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)
        
        print(f"\n{name} Results:")
        print(f"Best Parameters: {results[name]['best_params']}")
        print(f"Cross-validation Score: {results[name]['cv_mean_score']:.3f} (+/- {results[name]['cv_std']*2:.3f})")
        print(f"Test Set Performance:")
        print(f"  Accuracy:  {results[name]['accuracy']:.3f}")
        print(f"  Precision: {results[name]['precision']:.3f}")
        print(f"  Recall:    {results[name]['recall']:.3f}")
        print(f"  F1 Score:  {results[name]['f1']:.3f}")
    
    return results, confusion_matrices, best_models, cv_scores

def visualize_model_comparison(results, cv_scores):
    """
    Create detailed visualization of model performance comparison
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metrics_data = {model: [results[model][metric] for metric in metrics] 
                   for model in results.keys()}
    
    metrics_df = pd.DataFrame(metrics_data, index=metrics)
    metrics_df.plot(kind='bar', ax=ax1)
    ax1.set_title('Model Performance Metrics Comparison')
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.legend(title='Models', bbox_to_anchor=(1.05, 1))
    
    cv_data = []
    for model, scores in cv_scores.items():
        for score in scores:
            cv_data.append({'Model': model, 'CV Score': score})
    
    cv_df = pd.DataFrame(cv_data)
    sns.boxplot(data=cv_df, x='Model', y='CV Score', ax=ax2)
    ax2.set_title('Cross-validation Score Distribution')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('CV Score')
    
    plt.tight_layout()
    plt.show()

def visualize_best_model_results(results, confusion_matrices):
    """
    Visualize detailed results for the best performing model
    """
    best_model = max(results.items(), key=lambda x: x[1]['f1'])[0]
    best_cm = confusion_matrices[best_model]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for Best Model: {best_model}\nF1 Score: {results[best_model]["f1"]:.3f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    true_neg, false_pos, false_neg, true_pos = best_cm.ravel()
    total = np.sum(best_cm)
    
    plt.figtext(0.02, -0.15, 
                f"""Model Performance Details:
                • True Negatives (correct negative predictions): {true_neg} ({true_neg/total*100:.1f}%)
                • False Positives (incorrect positive predictions): {false_pos} ({false_pos/total*100:.1f}%)
                • False Negatives (incorrect negative predictions): {false_neg} ({false_neg/total*100:.1f}%)
                • True Positives (correct positive predictions): {true_pos} ({true_pos/total*100:.1f}%)""",
                fontsize=10, ha='left')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.show()

def print_statistical_comparison(results):
    """
    Print statistical comparison of model performances
    """
    print("\nDetailed Statistical Comparison:")
    print("-" * 50)
    
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"CV Score: {metrics['cv_mean_score']:.3f} ± {metrics['cv_std']*2:.3f}")
        print("Best Parameters:")
        for param, value in metrics['best_params'].items():
            print(f"  {param}: {value}")
    
    best_model = max(results.items(), key=lambda x: x[1]['cv_mean_score'])[0]
    print(f"\nBest Model: {best_model}")
    print(f"Best CV Score: {results[best_model]['cv_mean_score']:.3f}")
    print(f"Test Set F1 Score: {results[best_model]['f1']:.3f}")

def main():
   
    print("Loading and exploring data...")
    df = load_and_explore_data('lung_cancer_data.csv')
    
    print("\nGenerating data distribution visualizations...")
    visualize_data_distribution(df)
    
    print("\nPreprocessing data...")
    X_scaled, y = preprocess_data(df)
    
    print("\nAnalyzing feature importance...")
    feature_importance = analyze_feature_importance(X_scaled, y)
    
    print("\nTraining and evaluating models...")
    results, confusion_matrices, best_models, cv_scores = train_and_evaluate_models(X_scaled, y)
    
    print("\nGenerating model comparison visualizations...")
    visualize_model_comparison(results, cv_scores)
    
    print("\nGenerating best model visualizations...")
    visualize_best_model_results(results, confusion_matrices)
    
    print_statistical_comparison(results)
    
    return results, best_models, feature_importance

if __name__ == "__main__":
    results, best_models, feature_importance = main()