# Lung Cancer Analysis and Prediction Report
Generated on: 2024-12-22 19:43:46

## 1. Dataset Overview
- Total number of samples: 309
- Positive cases: 270
- Negative cases: 39
- Number of features: 15

### Gender Distribution
- Male: 162
- Female: 147

### Age Statistics
- Mean age: 62.67
- Minimum age: 21.0
- Maximum age: 87.0
- Standard deviation: 8.21

## 2. Risk Factor Analysis
### Top 5 Most Prevalent Risk Factors
- SHORTNESS OF BREATH: 164.1%
- COUGHING: 157.9%
- YELLOW_FINGERS: 157.0%
- SMOKING: 156.3%
- CHEST PAIN: 155.7%

## 3. Model Performance
### Model Metrics
- Accuracy: 0.935
- Precision: 0.983
- Recall: 0.950
- F1 Score: 0.966

### Best Model Parameters
- class_weight: balanced
- max_depth: None
- min_samples_split: 5
- n_estimators: 300

### Top 5 Most Important Features
- GENDER: 0.027
- AGE: 0.102
- SMOKING: 0.032
- YELLOW_FINGERS: 0.062
- ANXIETY: 0.048

## 4. Visualizations
The following visualizations have been generated and saved in the 'report_visualizations' directory:
1. Age Distribution by Lung Cancer Status
2. Risk Factors Correlation Matrix
3. Feature Importance Plot

## 5. Conclusions and Recommendations
1. Based on the analysis, the most significant risk factors for lung cancer prediction are:
   GENDER (0.027), AGE (0.102), SMOKING (0.032)

2. The model shows excellent performance with an F1 score of 0.966.

3. Key findings:
   - The most common risk factor is SHORTNESS OF BREATH with a prevalence of 164.1%

## 6. Model Usage Instructions
The trained model has been saved and can be used for making predictions using the provided prediction interface. For new predictions:
1. Use the standardized input format
2. Consider all risk factors
3. Ensure all required features are provided

Note: This model should be used as a screening tool only and not as a definitive diagnostic tool.
