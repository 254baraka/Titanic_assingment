# Titanic Survival Prediction

## Project Overview
This project implements machine learning models to predict passenger survival on the Titanic. Using passenger characteristics such as class, age, gender, and fare, we build and compare two classification algorithms: Logistic Regression and K-Nearest Neighbors (KNN).

## Dataset
The dataset contains information about Titanic passengers including:
- **Survival**: 0 = No, 1 = Yes (Target variable)
- **Pclass**: Passenger class (1st, 2nd, 3rd)
- **Sex**: Gender of passenger
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: Passenger fare
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

  titanic-survival-prediction/
│
├── titanic_survival.py          # Main script for prediction
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── outputs/                      # Generated outputs folder
    ├── confusion_matrices.png    # Model confusion matrices
    ├── feature_importance.png    # Feature importance plot
    └── knn_performance.png       # KNN accuracy plot

  1. Data Loading
python
import seaborn as sns
df = sns.load_dataset("titanic")

2. Data Preprocessing
- Handle missing values:

- Fill missing 'age' with median value

- Fill missing 'embarked' with mode

- Drop redundant columns ('deck', 'alive', 'who', 'adult_male', 'class', 'embark_town')

- Encode categorical variables:

- x: male=0, female=1

- Embarked: S=0, C=1, Q=2

- Feature scaling using StandardScaler

  3. Models Implemented
# Logistic Regression
- Linear model for binary classification

- Uses sigmoid function to predict probabilities

- Regularization applied to prevent overfitting

# K-Nearest Neighbors (KNN)
- Non-parametric classification algorithm

- Predicts based on k nearest neighbors

- Includes hyperparameter tuning for optimal k value

  4. Training Configuration
- Train-test split: 80% training, 20% testing

- Random state: 42 (for reproducibility)

- Stratified splitting to maintain class distribution

  # Output and Results
  # Model Performance Metrics

  The script outputs:

- Classification Report: Precision, Recall, F1-Score for each class

- Confusion Matrix: True positives, false positives, true negatives, false negatives

- Accuracy Score: Overall prediction accuracy

  # Visualizations Generated
- Confusion Matrices: Side-by-side comparison for both models

- Feature Importance: Logistic regression coefficient analysis

- KNN Performance: Accuracy vs. number of neighbors plot

Sample Output
============================================================
LOGISTIC REGRESSION RESULTS
============================================================
Classification Report:
              precision    recall  f1-score   support
Not Survived       0.82      0.84      0.83       110
    Survived       0.77      0.74      0.76        80

    accuracy                           0.80       190
   macro avg       0.79      0.79      0.79       190
weighted avg       0.80      0.80      0.80       190

Accuracy: 0.8000


# Key Findings
- Most Important Features (from Logistic Regression coefficients):

- Sex (female passengers have much higher survival probability)

- Passenger class (1st class passengers more likely to survive)

- Fare (higher fare correlates with higher survival)

- Age (younger passengers slightly more likely to survive)

# Model Performance:

- Logistic Regression generally outperforms KNN by 1-2%

-KNN requires careful selection of k (usually 5-11 works best)

- Both models achieve ~80% accuracy
