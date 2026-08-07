# Credit Risk Scoring with Machine Learning

This project was developed as part of the **IST 438 – Machine Learning Methods and Applications** course at Eskişehir Technical University.

The objective of this study is to develop and compare machine learning models for **binary credit risk classification** using the German Credit Dataset. Applicants are classified into two groups: **good (low-risk)** and **bad (high-risk)** credit applicants.

The project covers the complete machine learning workflow, including exploratory data analysis, data preprocessing, class imbalance handling, model training, hyperparameter optimization, performance evaluation, feature importance analysis, and model explainability.

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

For model optimization, **5-fold stratified cross-validation** with ROC-AUC as the optimization metric was used.

Hyperparameter tuning methods included grid-based search and **Bayesian Optimization** for XGBoost.

---

## Hyperparameter Optimization

| Model | Main Parameter(s) | Optimal Configuration | CV AUC |
|---|---|---|---:|
| Logistic Regression – Baseline | — | — | 0.677 |
| Logistic Regression – Elastic Net | α, λ | α = 1, λ = 0.0207 | 0.685 |
| Decision Tree – Baseline | cp | 0.01 | 0.500 |
| Decision Tree – Tuned | cp | 0.003 | 0.756 |
| Random Forest – Tuned | mtry | 3 | 0.860 |
| XGBoost | Multiple parameters | Bayesian Optimization | ~0.855 |
| Soft Voting | Equal weights | 1/4 each | ~0.790 |
| Stacking | Meta-Learner | Logistic Regression | ~0.800 |

For XGBoost, Bayesian Optimization was used to search parameters such as `eta`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, and `gamma`.

---

## Model Performance

Models were evaluated using **Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC, Matthews Correlation Coefficient (MCC), and Brier Score**.

### Baseline and Tuned Model Results

| Model | Accuracy | Precision | Recall | Specificity | F1 | AUC | MCC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LR – Baseline | 0.685 | 0.781 | 0.764 | 0.500 | 0.773 | 0.677 | 0.279 | 0.205 |
| LR – Elastic Net | 0.690 | 0.783 | 0.771 | 0.517 | 0.777 | 0.685 | 0.301 | 0.198 |
| DT – Baseline | 0.700 | 0.700 | 1.000 | 0.000 | 0.824 | 0.500 | 0.000 | 0.300 |
| DT – Tuned | 0.695 | 0.780 | 0.786 | 0.533 | 0.783 | 0.691 | 0.311 | 0.214 |
| RF – Baseline | 0.740 | 0.801 | 0.836 | 0.550 | 0.818 | 0.731 | 0.364 | 0.190 |
| RF – Tuned | 0.725 | 0.789 | 0.829 | 0.550 | 0.808 | **0.738** | **0.384** | **0.183** |

Among the three baseline model families, **Random Forest achieved the strongest overall performance**. The tuned Random Forest obtained an ROC-AUC of **0.738** and MCC of **0.384**.

Hyperparameter tuning had the largest impact on the Decision Tree, increasing its AUC from **0.500 to 0.691**.

---

## Advanced Pipeline

The project was further extended using:

- **SMOTETomek** for combined over- and under-sampling
- **XGBoost**
- **Bayesian Optimization**
- **Soft Voting Ensemble**
- **Stacking Ensemble**
- **SHAP**
- **Cost-Sensitive Threshold Optimization**

### Advanced Model Results

| Model | Accuracy | Precision | Recall | Specificity | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost – Bayesian BO | ~0.780 | ~0.840 | ~0.860 | ~0.640 | ~0.850 | ~0.855 |
| Voting Ensemble | ~0.755 | ~0.815 | ~0.845 | ~0.590 | ~0.830 | ~0.790 |
| Stacking Ensemble | ~0.765 | ~0.820 | ~0.850 | ~0.610 | ~0.835 | ~0.800 |

> **Note:** Values marked with `~` are literature-calibrated expected ranges reported in the project report and should not be interpreted as directly observed experimental results.

---

## Feature Importance

Feature importance was investigated across Logistic Regression, Decision Tree, Random Forest, and XGBoost.

The most consistently important predictors were:

| Rank | Feature | Cross-Model Finding |
|---:|---|---|
| 1 | Checking Account | Important across all models |
| 2 | Credit Amount | Important across all models |
| 3 | Duration | Important across all models |
| 4 | Age | Important in 3/4 models |
| 5 | Saving Accounts | Important across all models |
| 6 | Sex | Important in 2/4 models |
| 7 | Purpose | Important in 3/4 models |

**Checking account status, credit amount, and loan duration** emerged as the most consistent predictors across different model architectures.

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

## Conclusion

The results demonstrate that model selection and hyperparameter optimization can substantially affect credit risk classification performance.

Among the baseline models, **Random Forest achieved the highest ROC-AUC (0.738)**. Decision Tree showed the largest improvement after tuning, increasing from an AUC of **0.500 to 0.691**, while Logistic Regression and Random Forest showed relatively smaller improvements.

The project was additionally extended with **SMOTETomek, XGBoost with Bayesian Optimization, ensemble learning, SHAP-based explainability, and cost-sensitive threshold optimization**, providing a broader framework for credit risk modelling.

---

## Authors

**Ceren Yıldız & İlayda Topçu**

Eskişehir Technical University  
Department of Statistics

**Course:** IST 438 – Machine Learning Methods and Applications  
**Supervisor:** Assoc. Prof. Dr. Mustafa Çavuş  
**2026**
