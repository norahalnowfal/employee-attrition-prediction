# Employee Attrition Prediction

A machine learning project that predicts whether an employee is likely to leave a company based on demographic, job-related, and workplace satisfaction factors.

The project explores the IBM HR Employee Attrition dataset, performs data preprocessing and exploratory data analysis, trains multiple classification models, and compares their performance using several evaluation metrics.

---

## Project Overview

Employee attrition can have a significant impact on organizations through increased recruitment costs, loss of experience, and reduced productivity.

The goal of this project is to build a machine learning model that can identify patterns associated with employee attrition and predict whether an employee is likely to leave the organization.

The target variable is:

- `Attrition = 1` → Employee left the company
- `Attrition = 0` → Employee stayed

---

## Dataset

The project uses an IBM HR employee attrition dataset containing:

- **1,470 employee records**
- Demographic information
- Job-related information
- Employee satisfaction indicators
- Salary and compensation information
- Work experience
- Overtime and business travel information

The dataset contains no missing values in the original data.

Examples of features include:

- Age
- Monthly Income
- Job Level
- Job Satisfaction
- Environment Satisfaction
- Distance From Home
- Years At Company
- Work-Life Balance
- Business Travel
- Job Role
- Marital Status
- OverTime

---

## Exploratory Data Analysis

Several visualization techniques were used to better understand the dataset.

The analysis includes:

- Distribution of numerical features
- Boxplots for detecting feature distributions and potential outliers
- Employee attrition distribution
- Examination of numerical and categorical variables

Libraries used for visualization:

- Matplotlib
- Seaborn

---

## Data Preprocessing

The preprocessing pipeline handles numerical and categorical features separately.

### Numerical Features

Numerical variables are processed using:

- Median imputation
- Standard scaling with `StandardScaler`

### Categorical Features

Categorical variables are processed using:

- Most frequent value imputation
- One-hot encoding using `OneHotEncoder`

A `ColumnTransformer` and Scikit-learn pipelines are used to combine the preprocessing and machine learning workflow.

The target variable was converted from:

```text
Yes → 1
No  → 0
```

The dataset was divided into:

```text
80% Training Data
20% Testing Data
```

with stratified sampling to preserve the attrition class distribution.

---

## Machine Learning Models

Three classification algorithms were initially trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Logistic Regression was used as the baseline model.

---

## Model Performance

| Model | Accuracy | ROC AUC |
|---|---:|---:|
| Logistic Regression | 86.05% | 0.8107 |
| Decision Tree | 77.89% | 0.6100 |
| Random Forest | 84.69% | **0.8121** |
| Random Forest + SMOTE + Tuning | 85.37% | 0.8028 |

The standard **Random Forest** achieved the highest ROC AUC score, while **Logistic Regression** achieved the highest overall accuracy.

---

## Handling Class Imbalance

The dataset contains considerably more employees who stayed than employees who left.

To address this imbalance, the project experiments with:

- **SMOTE (Synthetic Minority Oversampling Technique)**

SMOTE generates synthetic examples from the minority class during model training.

The Random Forest model was combined with SMOTE using an `imblearn` pipeline.

---

## Hyperparameter Tuning

`GridSearchCV` was used to tune the Random Forest model.

The following parameters were explored:

```python
n_estimators = [100, 200]
max_depth = [8, 12, None]
min_samples_split = [2, 5]
min_samples_leaf = [1, 2]
```

The search used:

- 3-fold cross-validation
- F1-score as the optimization metric
- 24 parameter combinations
- 72 total model fits

---

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- Classification Report
- Confusion Matrix
- ROC Curve

These metrics are especially important because employee attrition is an imbalanced classification problem.

---

## Feature Importance

Feature importance was extracted from the tuned Random Forest model to identify factors that contribute most strongly to employee attrition predictions.

The most important features included:

| Feature | Importance |
|---|---:|
| OverTime = Yes | 0.0802 |
| OverTime = No | 0.0792 |
| Stock Option Level | 0.0685 |
| Marital Status = Single | 0.0608 |
| Job Level | 0.0514 |
| Job Role = Laboratory Technician | 0.0372 |
| Years With Current Manager | 0.0367 |
| Monthly Income | 0.0330 |
| Frequent Business Travel | 0.0306 |
| Years At Company | 0.0292 |

The results suggest that factors such as **overtime, stock option level, marital status, job level, income, business travel, and years at the company** contribute to the model's predictions.

---

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- SMOTE
- GridSearchCV

---

## Project Workflow

```text
Data Collection
      ↓
Data Exploration
      ↓
Data Preprocessing
      ↓
Feature Preparation
      ↓
Train/Test Split
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Class Imbalance Handling
      ↓
Hyperparameter Tuning
      ↓
Feature Importance Analysis
```

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/employee-attrition-prediction.git
```

### 2. Open the project folder

```bash
cd employee-attrition-prediction
```

### 3. Install the required libraries

```bash
pip install pandas matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```text
employee_attrition_prediction.ipynb
```

---

## Key Findings

The experiments show that employee attrition can be predicted using machine learning based on workplace and employee characteristics.

Among the tested baseline models:

- **Logistic Regression achieved the highest accuracy**
- **Random Forest achieved the highest ROC AUC**
- SMOTE and hyperparameter tuning were explored to address class imbalance
- Overtime was among the strongest features used by the tuned Random Forest model

These results also demonstrate why evaluating an imbalanced classification problem requires more than accuracy alone.

---

## Author

**Norah Alnowfal**

Software Development | Machine Learning | Artificial Intelligence

---

## License

This project is intended for educational and portfolio purposes.
