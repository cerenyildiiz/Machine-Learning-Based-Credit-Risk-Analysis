# Credit Risk Scoring with Machine Learning

This project was developed as part of the **IST 438 – Machine Learning Methods and Applications** course at Eskişehir Technical University.

The objective of this study is to develop and compare machine learning models for **binary credit risk classification** using the German Credit Dataset. Applicants are classified into two groups: **good (low-risk)** and **bad (high-risk)** credit applicants.

The project covers the complete machine learning workflow, including exploratory data analysis, data preprocessing, class imbalance handling, model training, hyperparameter optimization, performance evaluation, feature importance analysis, ensemble learning, and model explainability.

---

## Dataset

The **German Credit Dataset** contains **1,000 loan applications** and a binary target variable representing credit risk.

| Property | Description |
|---|---|
| Dataset | German Credit Dataset |
| Observations | 1,000 |
| Target | Risk |
| Classes | Good / Bad |
| Good Risk | 700 (70%) |
| Bad Risk | 300 (30%) |
| Train / Test Split | 80% / 20% |

The dataset contains numerical and categorical information such as age, credit amount, loan duration, checking account status, saving account status, housing, job, and loan purpose.
<img width="3000" height="2100" alt="image" src="https://github.com/user-attachments/assets/1db8055c-a7b3-4318-a919-55156c7d7360" />

---

## Data Preprocessing

Several preprocessing steps were applied before model development:

- Missing values in `Saving_accounts` and `Checking_account` were represented as an explicit **unknown** category.
- Binary, ordinal, and one-hot encoding were applied according to variable type.
- The dataset was divided using an **80/20 stratified train-test split**.
- Numerical variables were standardized using **Z-score standardization**.
- Standardization parameters were estimated only from the training set to prevent data leakage.
- **SMOTE** was applied only to the training data to address class imbalance.

After SMOTE, the minority `bad` class increased from **240 to 448 observations**, while the `good` class remained at **560**.

---

## Exploratory Data Analysis

Exploratory analysis indicated several important relationships with credit risk.

| Feature / Relationship | Finding |
|---|---|
| Credit Amount ↔ Duration | Strong positive relationship (r = 0.62) |
| Checking Account ↔ Risk | r = -0.20 |
| Duration ↔ Risk | r = -0.21 |
| Credit Amount ↔ Risk | r = -0.15 |
| Age ↔ Risk | r = 0.09 |
| Sex ↔ Risk | r = 0.08 |

Longer loan durations and higher credit amounts were associated with higher credit risk. **Checking account status** was also identified as an important categorical discriminator.

---

## Machine Learning Models

The following classification algorithms were investigated:

- **Logistic Regression**
- **Decision Tree (CART)**
- **Random Forest**
- **XGBoost**
- **Soft Voting Ensemble**
- **Stacking Ensemble**

Baseline and hyperparameter-tuned configurations were considered during model development.

For model optimization, **5-fold stratified cross-validation** with ROC-AUC as the optimization metric was used.

---

## Hyperparameter Optimization

Different optimization strategies were applied depending on the machine learning algorithm.

| Model | Main Parameter(s) | Optimization |
|---|---|---|
| Logistic Regression | α, λ | Elastic Net |
| Decision Tree | cp | Grid Search |
| Random Forest | mtry | Grid Search |
| XGBoost | eta, max_depth, min_child_weight, subsample, colsample_bytree, gamma | Bayesian Optimization |

The selected configurations for the models with directly observed results were:

- **Logistic Regression:** α = 1, λ = 0.020691
- **Decision Tree:** cp = 0.003
- **Random Forest:** mtry = 3

For XGBoost, **Bayesian Optimization** was explored to efficiently search across multiple hyperparameters.

---

## Model Performance

Models with directly observed test-set results were evaluated using **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**.

### Baseline and Tuned Model Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression – Baseline | 0.685 | 0.781 | 0.764 | 0.773 | 0.677 |
| Logistic Regression – Elastic Net | 0.690 | 0.783 | 0.771 | 0.777 | 0.685 |
| Decision Tree – Baseline | 0.700 | 0.700 | 1.000 | 0.824 | 0.500 |
| Decision Tree – Tuned | 0.695 | 0.780 | 0.786 | 0.783 | 0.691 |
| Random Forest – Baseline | **0.740** | **0.801** | **0.836** | **0.818** | 0.731 |
| Random Forest – Tuned | 0.725 | 0.789 | 0.829 | 0.808 | **0.738** |

Among the models with directly observed test results, **Random Forest achieved the strongest overall performance**.

The baseline Random Forest achieved the highest **Accuracy (0.740)**, while the tuned Random Forest achieved the highest **ROC-AUC (0.738)**.

Hyperparameter tuning had the largest impact on the Decision Tree, increasing its ROC-AUC from **0.500 to 0.691**.

---

## Hyperparameter Tuning Gains

| Model | Baseline AUC | Tuned AUC | Δ AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.677 | 0.685 | +0.008 |
| Decision Tree | 0.500 | 0.691 | **+0.191** |
| Random Forest | 0.731 | 0.738 | +0.007 |

The **Decision Tree** showed the largest improvement after hyperparameter tuning, while Logistic Regression and Random Forest showed relatively smaller gains.

---

## Advanced Pipeline

The project was further extended using:

- **SMOTETomek** for combined over- and under-sampling
- **XGBoost**
- **Bayesian Optimization**
- **Soft Voting Ensemble**
- **Stacking Ensemble**
- **SHAP (SHapley Additive exPlanations)**
- **Cost-Sensitive Threshold Optimization**

These extensions were explored to improve class balance, predictive performance, model interpretability, and decision-making under asymmetric classification costs.

### SMOTETomek

In addition to SMOTE, **SMOTETomek** was considered as an advanced resampling strategy.

SMOTE generates synthetic observations for the minority class, while Tomek Links remove ambiguous observations near class boundaries. Combining these approaches can produce a cleaner and more balanced training set.

### Ensemble Learning

Two ensemble strategies were explored:

- **Soft Voting:** combines predicted probabilities from multiple classifiers.
- **Stacking:** combines predictions from base learners through a meta-learner.

These approaches were investigated to determine whether combining different model structures could improve predictive performance and robustness.

---

## Feature Importance

Feature importance analysis was used to investigate which variables contributed most strongly to credit risk classification.

For the **tuned Random Forest**, feature importance was evaluated using the Mean Decrease in Gini Impurity.

Important predictors included variables related to:

- **Checking Account**
- **Credit Amount**
- **Duration**
- **Age**
- **Saving Accounts**

The results suggest that applicants' financial status, credit characteristics, and loan duration contain important information for distinguishing between good- and bad-risk applicants.

---

## Explainable AI

**SHAP (SHapley Additive exPlanations)** was incorporated into the advanced pipeline to improve model interpretability.

SHAP analysis was used for:

- Global feature importance
- Feature directionality
- Feature dependence analysis
- Individual applicant-level explanations

This allows model predictions to be interpreted not only in terms of predictive performance but also in terms of how individual variables contribute to credit risk predictions.

---

## Cost-Sensitive Threshold Optimization

In credit risk classification, different prediction errors may have different financial consequences.

A **cost-sensitive threshold optimization** approach was therefore considered instead of relying exclusively on the default classification threshold of 0.50.

The approach assigns different costs to:

- **False Positives:** rejecting a good applicant
- **False Negatives:** approving a bad applicant

Since approving a high-risk applicant may result in greater financial loss, the classification threshold can be adjusted according to the relative costs of these errors.

---

## Conclusion

The results demonstrate that model selection, class imbalance handling, and hyperparameter optimization can substantially affect credit risk classification performance.

Among the models with directly observed test-set results, **Random Forest achieved the strongest overall performance**. The baseline Random Forest achieved the highest accuracy of **0.740**, while the tuned Random Forest achieved the highest ROC-AUC of **0.738**.

Decision Tree showed the largest improvement after tuning, increasing its ROC-AUC from **0.500 to 0.691**, while Logistic Regression and Random Forest showed relatively smaller improvements.

The project was further extended with **SMOTETomek, XGBoost with Bayesian Optimization, ensemble learning, SHAP-based explainability, and cost-sensitive threshold optimization**, providing a broader framework for credit risk modelling.

---

## Authors

**Ceren Yıldız & İlayda Topçu**

Eskişehir Technical University  
Department of Statistics

**Course:** IST 438 – Machine Learning Methods and Applications  
**Supervisor:** Assoc. Prof. Dr. Mustafa Çavuş  
**2026**
