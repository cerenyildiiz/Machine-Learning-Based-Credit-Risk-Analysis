# Credit Risk Scoring with Machine Learning

This project was developed as part of the **IST 438 – Machine Learning Methods and Applications** course at Eskişehir Technical University.

The objective of this study is to develop and compare machine learning models for **binary credit risk classification** using the German Credit Dataset. Loan applicants are classified into two groups: **good (low-risk)** and **bad (high-risk)** credit applicants.

The project covers the machine learning workflow from exploratory data analysis and preprocessing to class imbalance handling, model training, hyperparameter optimization, performance evaluation, and feature importance analysis.

---

## Dataset

The **German Credit Dataset** contains **1,000 loan applications** with a binary target variable representing credit risk.

| Property | Description |
|---|---|
| Dataset | German Credit Dataset |
| Observations | 1,000 |
| Target | Risk |
| Classes | Good / Bad |
| Good Risk | 700 (70%) |
| Bad Risk | 300 (30%) |
| Train / Test Split | 80% / 20% |

The dataset includes numerical and categorical variables such as **age, credit amount, loan duration, checking account status, saving account status, housing, job, and loan purpose**.

---

## Exploratory Data Analysis

Exploratory data analysis was performed to investigate the distribution of credit risk and the relationships between predictors and the target variable.

The dataset contains a moderate class imbalance, with **70% good-risk** and **30% bad-risk** applicants.

Several relationships were observed:

| Feature / Relationship | Correlation |
|---|---:|
| Credit Amount ↔ Duration | 0.62 |
| Checking Account ↔ Risk | -0.20 |
| Duration ↔ Risk | -0.21 |
| Credit Amount ↔ Risk | -0.15 |
| Age ↔ Risk | 0.09 |
| Sex ↔ Risk | 0.08 |

Credit amount and loan duration showed the strongest relationship among the predictors. Longer loan durations and higher credit amounts were generally associated with higher credit risk.

**Checking account status** was also identified as an important categorical discriminator between good- and bad-risk applicants.

---

## Data Preprocessing

Several preprocessing steps were applied before model development:

- Missing values in `Saving_accounts` and `Checking_account` were represented as an explicit **unknown** category.
- Binary, ordinal, and one-hot encoding were applied according to variable type.
- The dataset was divided using an **80/20 stratified train-test split**.
- Numerical variables were standardized using **Z-score standardization**.
- Scaling parameters were estimated only from the training set to prevent data leakage.
- **SMOTE** was applied only to the training data to address class imbalance.

### Class Distribution After SMOTE

| Class | Before SMOTE | After SMOTE |
|---|---:|---:|
| Bad | 240 | 448 |
| Good | 560 | 560 |

SMOTE increased the representation of the minority `bad` class while leaving the test set unchanged.

---

## Machine Learning Models

The following classification algorithms were investigated:

- **Logistic Regression**
- **Decision Tree (CART)**
- **Random Forest**
- **XGBoost**

Baseline and hyperparameter-tuned configurations were considered during model development.

For model optimization, **5-fold stratified cross-validation** with **ROC-AUC** as the optimization metric was used.

---

## Hyperparameter Optimization

Different optimization approaches were considered depending on the machine learning algorithm.

| Model | Optimization | Main Parameters | Selected Configuration |
|---|---|---|---|
| Logistic Regression | Elastic Net | α, λ | α = 1, λ = 0.020691 |
| Decision Tree | Grid Search | cp | cp = 0.003 |
| Random Forest | Grid Search | mtry | mtry = 3 |
| XGBoost | Bayesian Optimization | Multiple parameters | — |

For Logistic Regression, **Elastic Net regularization** was used to control model complexity and perform coefficient shrinkage.

For Decision Tree and Random Forest, grid search was used to determine the optimal `cp` and `mtry` values, respectively.

**Bayesian Optimization** was explored for tuning XGBoost hyperparameters.

---

## Model Performance

Models with directly observed test-set results were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### Baseline and Tuned Model Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression – Baseline | 0.685 | 0.781 | 0.764 | 0.773 | 0.677 |
| Logistic Regression – Tuned | 0.690 | 0.783 | 0.771 | 0.777 | 0.685 |
| Decision Tree – Baseline | 0.700 | 0.700 | 1.000 | 0.824 | 0.500 |
| Decision Tree – Tuned | 0.695 | 0.780 | 0.786 | 0.783 | 0.691 |
| Random Forest – Baseline | **0.740** | **0.801** | **0.836** | **0.818** | 0.731 |
| Random Forest – Tuned | 0.725 | 0.789 | 0.829 | 0.808 | **0.738** |

Among the models with directly observed test results, **Random Forest showed the strongest overall performance**.

The baseline Random Forest achieved the highest **Accuracy (0.740)**, while the tuned Random Forest achieved the highest **ROC-AUC (0.738)**.

---

## Hyperparameter Tuning Gains

The effect of hyperparameter tuning was also examined by comparing baseline and tuned ROC-AUC values.

| Model | Baseline AUC | Tuned AUC | Δ AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.677 | 0.685 | +0.008 |
| Decision Tree | 0.500 | 0.691 | **+0.191** |
| Random Forest | 0.731 | 0.738 | +0.007 |

The largest improvement was observed for the **Decision Tree**, where tuning increased ROC-AUC from **0.500 to 0.691**.

Logistic Regression and Random Forest showed smaller improvements after hyperparameter optimization.

---

## Best Model

Based on ROC-AUC performance, the **tuned Random Forest** was selected as the best-performing model.

| Metric | Value |
|---|---:|
| Accuracy | 0.725 |
| Precision | 0.789 |
| Recall | 0.829 |
| F1-Score | 0.808 |
| ROC-AUC | **0.738** |

### Confusion Matrix

The tuned Random Forest produced the following classification results on the test set:

|  | Actual Bad | Actual Good |
|---|---:|---:|
| **Predicted Bad** | 29 | 24 |
| **Predicted Good** | 31 | 116 |

The model correctly classified **116 good-risk applicants** and **29 bad-risk applicants**.

---

## ROC Analysis

ROC curves were used to compare the discrimination ability of the tuned models.

| Tuned Model | ROC-AUC |
|---|---:|
| Logistic Regression | 0.685 |
| Decision Tree | 0.691 |
| Random Forest | **0.738** |

The tuned Random Forest achieved the highest ROC-AUC among the evaluated models.

---

## Feature Importance

Feature importance analysis was performed using the **tuned Random Forest model**.

The analysis showed that variables related to **checking account status, credit amount, loan duration, age, and saving account status** played important roles in credit risk classification.

Feature importance provides additional insight into which applicant characteristics contribute most strongly to model predictions.

---

## Optimal Decision Threshold

In addition to model comparison, an **optimal decision threshold** was investigated.

Instead of relying only on the default classification threshold of 0.50, threshold analysis can be used to examine the trade-off between correctly identifying good- and bad-risk applicants.

This is particularly relevant in credit risk applications, where different classification errors may have different financial consequences.

---

## Conclusion

This project demonstrates a complete machine learning workflow for **credit risk classification**, including exploratory data analysis, preprocessing, class imbalance handling, model development, hyperparameter tuning, and performance evaluation.

Among the models with directly observed test-set results, **Random Forest achieved the strongest overall performance**. The baseline Random Forest obtained the highest accuracy of **0.740**, while the tuned Random Forest achieved the highest ROC-AUC of **0.738**.

Hyperparameter tuning had the largest impact on the Decision Tree, improving its ROC-AUC from **0.500 to 0.691**.

Overall, the study highlights the importance of **class imbalance handling, cross-validation, hyperparameter optimization, model comparison, ROC analysis, and feature importance** when developing machine learning models for credit risk assessment.

---

## Authors

**Ceren Yıldız & İlayda Topçu**

Eskişehir Technical University  
Department of Statistics

**Course:** IST 438 – Machine Learning Methods and Applications  
**Supervisor:** Assoc. Prof. Dr. Mustafa Çavuş  
**2026**
