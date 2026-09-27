# Module 2: Analytics & Predictive Modeling Pipeline (/analytics)

## Overview
This module demonstrates an end-to-end data science and machine learning workflow on the classic Titanic customer-style dataset. The dataset is loaded once, saved as a committed offline fallback (`titanic.csv`), profiled, cleaned, visualized, and used to train and evaluate leakage-free predictive pipelines.

---

## Part A: Profiling, Cleaning, and Data Story

### 1. Missing Value Strategy & Threshold Rule
The dataset contains 891 records. Missing percentages were measured as:
- **`embarked` / `embark_town` (0.22% missing, 2 rows)**: Falls in the **< 5%** threshold. Dropped these 2 rows. With 99.78% data retention, dropping prevents introducing synthetic noise into location data.
- **`age` (19.87% missing, 177 rows)**: Falls in the **5%–30%** threshold. Imputed using the **median age (28.0 years)**. Median is preferred over mean due to the presence of age outliers.
- **`deck` (77.22% missing, 688 rows)**: Falls in the **> 30%** threshold. Missingness is excessively high for reliable imputation (over three-quarters fabricated data). Imputation would introduce severe noise and cardinality sparsity, so the column was **dropped**.
- Redundant columns (`alive`, `embark_town`, `adult_male`, `alone`, `class`, `who`) were removed to prevent collinearity.

### 2. Univariate Distribution & Outlier Analysis (IQR Rule)
Using the Interquartile Range (IQR) rule $[Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}]$:
- **Age**: $Q1 = 22.00$, $Q3 = 35.00$, $\text{IQR} = 13.00$. Valid range: $[2.50, 54.50]$. Outliers: **65 rows (7.31%)**.
- **Fare**: $Q1 = 7.90$, $Q3 = 31.00$, $\text{IQR} = 23.10$. Valid range: $[-26.76, 65.66]$. Outliers: **114 rows (12.82%)**.

**Fare Skewness:**
- **Mean:** 32.10
- **Median:** 14.45
- **Mode:** 8.05
- **Ordering:** $\text{Mean (32.10)} > \text{Median (14.45)} > \text{Mode (8.05)}$.
- **Conclusion:** Fare exhibits strong **right-skewness (positive skew)** caused by a long tail of high-value first-class tickets.

### 3. Bivariate Analysis (Boolean Masking)
- **Survival by Sex:**
  - Female: **74.04%**
  - Male: **18.89%**
- **Survival by Class:**
  - Class 1: **62.62%**
  - Class 2: **47.28%**
  - Class 3: **24.24%**
- **Survival by Sex and Class:**
  - Class 1 Female: **96.74%** | Class 1 Male: **36.89%**
  - Class 2 Female: **92.11%** | Class 2 Male: **15.74%**
  - Class 3 Female: **50.00%** | Class 3 Male: **13.54%**

### 4. Correlation Analysis (6 Numeric Features)
Correlation matrix computed strictly across `survived`, `pclass`, `age`, `sibsp`, `parch`, and `fare`:
1. **`pclass` vs `fare` ($r = -0.548$)**: Strong negative correlation reflecting ticket economics; higher social classes (lower numeric pclass 1) paid substantially higher fares.
2. **`sibsp` vs `parch` ($r = 0.415$)**: Moderate positive correlation reflecting family unit travel; passengers with siblings/spouses were also likely traveling with parents/children.

### 5. Multivariate Data Story
Four supporting visualizations were generated and saved in `analytics/plots/`:
1. `03_survival_sex_class.png`: Confirms survival was strictly prioritized by gender and economic tier; 1st class females had near-certain survival (96.74%), whereas 3rd class males suffered an 86.46% mortality rate.
2. `04_fare_pclass_survival.png`: Violin plot displaying that higher fare distributions directly mapped to elevated survival outcomes across all ticket classes.
3. `05_survivor_age_density.png`: Age density shows a survival spike among young male children, whereas adult males between 20–40 bore the brunt of casualties.
4. `06_family_size_survival.png`: Moderate family sizes (2 to 4 members) achieved optimal survival (~55–70%), whereas solo travelers and very large families (5+) experienced steep drop-offs in survival.

### 6. Exploratory Standardization Check
Applying $z = \frac{x - \mu}{\sigma}$ across the full dataset verified:
- **Age Z-Score**: Mean = 0.0000, Std = 1.0000
- **Fare Z-Score**: Mean = 0.0000, Std = 1.0000

---

## Part B: Predictive Modeling & Pipeline Engineering

### 1. Stratified Train/Test Split
- **Class Balance**: Deceased = 549 (61.8%), Survived = 340 (38.2%).
- **Justification**: A stratified split (80/20) preserves this exact 61.8% to 38.2% ratio in both training (711 rows) and testing (178 rows) sets, ensuring the evaluation metric is not skewed by sampling variance.

### 2. Leakage-Free Preprocessing Pipeline
Enforced via `ColumnTransformer` inside a `Pipeline`:
- Numeric features (`age`, `fare`, `sibsp`, `parch`): `SimpleImputer(strategy='median')` + `StandardScaler()`.
- Categorical features (`sex`, `embarked`, `pclass`): `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(drop='first')`.
- All steps fit strictly on `X_train` and applied in transform-only mode to `X_test`.

### 3. Classifier Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | AUC | Confusion Matrix [TN, FP, FN, TP] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 81.46% | 80.70% | 67.65% | 0.7360 | 0.8582 | [99, 11, 22, 46] |
| **Decision Tree** | 80.34% | 78.95% | 66.18% | 0.7200 | 0.8131 | [98, 12, 23, 45] |
| **Random Forest** | **83.71%** | **88.24%** | 66.18% | **0.7563** | **0.8512** | [104, 6, 23, 45] |

### 4. Imbalance Handling Comparison (Random Forest)

| Strategy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: |
| **Baseline (Unweighted)** | **88.24%** | 66.18% | **0.7563** |
| **class_weight='balanced'** | 76.92% | **73.53%** | 0.7519 |
| **SMOTE (Train fold only)** | 80.70% | 67.65% | 0.7360 |

**Conclusion on Imbalance**:
The baseline Random Forest achieves the highest overall precision (88.24%) and F1-score (0.7563) with minimal false positives (only 6 false positives). However, `class_weight='balanced'` successfully boosted recall from 66.18% to 73.53% by penalizing minority misclassifications, making it preferable if false negatives carry higher business penalty. SMOTE provided a middle ground but did not outperform `class_weight='balanced'`.

### 5. Hyperparameter Tuning & Out-of-Bag (OOB) Score
`GridSearchCV` over 5-fold cross-validation yielded:
- **Best Parameters**: `max_depth = 6`, `max_features = 'sqrt'`, `n_estimators = 150`.
- **Out-Of-Bag (OOB) Score**: **0.8256** (82.56% out-of-bag accuracy on unseen bootstrapping folds).

### 6. Regression Side-Task: Predicting Fare
Multivariate Linear Regression predicting continuous ticket `fare`:
- **MAE**: 17.88
- **RMSE**: 40.49
- **$R^2$**: 0.3854
- **Adjusted $R^2$**: 0.3601
- **Heteroscedasticity Analysis**: The residual plot (`09_regression_residuals.png`) displays an expanding cone/funnel shape as predicted fares increase. This confirms **heteroscedasticity**, indicating that while economy fares are tightly bounded, luxury fares have high variance driven by unobserved factors.

### 7. Final Model Comparison & Deployment Recommendation

| Metric Category | Metric | Value |
| :--- | :--- | :---: |
| **Classification (Random Forest)** | Accuracy | **83.71%** |
| | Precision | **88.24%** |
| | Recall | **66.18%** |
| | F1 Score | **0.7563** |
| | ROC AUC | **0.8512** |
| **Regression (Linear Regression)** | MAE | 17.88 |
| | RMSE | 40.49 |
| | $R^2$ | 0.3854 |
| | Adjusted $R^2$ | 0.3601 |

**Deployment Recommendation**:
I recommend deploying the **Random Forest Classifier** tuned with 150 estimators and a maximum depth of 6. It achieved the highest overall accuracy (83.71%), best-in-class precision (88.24%), and an AUC of 0.8512, producing only 6 false positives on the test set. Its robust Out-Of-Bag generalization score (82.56%) confirms stability against overfitting compared to the single Decision Tree and linear boundaries of Logistic Regression.

---

## How to Run

```bash
# 1. Run Data Profiling & Fallback Verification
python analytics/01_profile.py

# 2. Run Exploratory Data Analysis & Generate Visualizations
python analytics/02_eda.py

# 3. Train Models, Run Hyperparameter Tuning, and Verify Joblib Reload
python analytics/03_modeling.py
