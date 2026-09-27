"""
Zepto Analytics Pipeline - Part B: Predictive Modeling, Tuning, and Pipeline Persistence
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

os.makedirs("analytics/plots", exist_ok=True)

# 1. Load the shared committed dataset
df = pd.read_csv("analytics/titanic.csv")

# Drop redundant and high-missing columns as established in Part A
df_model = df.drop(columns=["deck", "embark_town", "alive", "adult_male", "alone", "class", "who"]).copy()
df_model = df_model.dropna(subset=["embarked"]).reset_index(drop=True)

print("==================================================")
print("1. STRATIFIED TRAIN/TEST SPLIT")
print("==================================================")
X = df_model.drop(columns=["survived"])
y = df_model["survived"]

n_total = len(y)
n_surv = int(y.sum())
n_dead = n_total - n_surv
print(f"Class Balance: Deceased = {n_dead} ({n_dead/n_total*100:.1f}%), Survived = {n_surv} ({n_surv/n_total*100:.1f}%)")
print("Justification: Stratified sampling preserves this 61.8% / 38.2% ratio across both train and test folds.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

print("\n==================================================")
print("2. LEAKAGE-FREE PREPROCESSING PIPELINE")
print("==================================================")
num_features = ("age", "fare", "sibsp", "parch")
cat_features = ("sex", "embarked", "pclass")

num_transformer = Pipeline(steps=(
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
))

cat_transformer = Pipeline(steps=(
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
))

preprocessor = ColumnTransformer(transformers=(
    ("num", num_transformer, list(num_features)),
    ("cat", cat_transformer, list(cat_features))
))

print("\n==================================================")
print("3. CLASSIFIER TRAINING & EVALUATION")
print("==================================================")
classifiers = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=4),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5)
}

results = []
trained_pipelines = {}

plt.figure(figsize=(8, 6))

for name, clf in classifiers.items():
    pipe = Pipeline(steps=(
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ))
    
    # Fit strictly on train fold
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe
    
    # Predict on test fold
    y_pred = pipe.predict(X_test)
    # Extract positive class probability
    y_prob = np.array([prob[1] for prob in pipe.predict_proba(X_test)])
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "AUC": auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot((0, 1), (0, 1), "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("analytics/plots/07_roc_curves.png", dpi=300)
plt.close()

df_clf_results = pd.DataFrame(results)
print(df_clf_results.to_string(index=False))

# Plot Decision Tree
dt_pipe = trained_pipelines["Decision Tree"]
dt_model = dt_pipe.named_steps["classifier"]
cat_encoder = dt_pipe.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
encoded_cat_names = cat_encoder.get_feature_names_out(list(cat_features))
all_feature_names = list(num_features) + list(encoded_cat_names)

plt.figure(figsize=(16, 8))
plot_tree(
    dt_model,
    feature_names=all_feature_names,
    class_names=("Died", "Survived"),
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Decision Tree Visualization (Max Depth = 4)")
plt.tight_layout()
plt.savefig("analytics/plots/08_decision_tree.png", dpi=300)
plt.close()
print("\nSaved ROC curve and Decision Tree plot to analytics/plots/")

print("\n==================================================")
print("4. IMBALANCE HANDLING COMPARISON (RANDOM FOREST)")
print("==================================================")
# (a) Baseline: Already computed
# (b) class_weight='balanced'
rf_balanced = Pipeline(steps=(
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5, class_weight="balanced"))
))
rf_balanced.fit(X_train, y_train)
y_pred_bal = rf_balanced.predict(X_test)

# (c) SMOTE applied to training data only
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_trans, y_train)

rf_smote = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5)
rf_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = rf_smote.predict(X_test_trans)

imbalance_comp = [
    {
        "Strategy": "Baseline (Unweighted)",
        "Precision": precision_score(y_test, trained_pipelines["Random Forest"].predict(X_test)),
        "Recall": recall_score(y_test, trained_pipelines["Random Forest"].predict(X_test)),
        "F1 Score": f1_score(y_test, trained_pipelines["Random Forest"].predict(X_test))
    },
    {
        "Strategy": "class_weight='balanced'",
        "Precision": precision_score(y_test, y_pred_bal),
        "Recall": recall_score(y_test, y_pred_bal),
        "F1 Score": f1_score(y_test, y_pred_bal)
    },
    {
        "Strategy": "SMOTE (Train fold only)",
        "Precision": precision_score(y_test, y_pred_smote),
        "Recall": recall_score(y_test, y_pred_smote),
        "F1 Score": f1_score(y_test, y_pred_smote)
    }
]
df_imbalance = pd.DataFrame(imbalance_comp)
print(df_imbalance.to_string(index=False))

print("\n==================================================")
print("5. HYPERPARAMETER TUNING & OUT-OF-BAG (OOB) SCORE")
print("==================================================")
rf_base = RandomForestClassifier(oob_score=True, random_state=42)
param_grid = {
    "classifier__n_estimators": (50, 100, 150),
    "classifier__max_depth": (4, 6, 8),
    "classifier__max_features": ("sqrt", "log2")
}
rf_tune_pipe = Pipeline(steps=(
    ("preprocessor", preprocessor),
    ("classifier", rf_base)
))
grid_search = GridSearchCV(rf_tune_pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf_pipe = grid_search.best_estimator_
best_oob_score = best_rf_pipe.named_steps["classifier"].oob_score_
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Corresponding Out-Of-Bag (OOB) Score: {best_oob_score:.4f}")

print("\n==================================================")
print("6. REGRESSION SIDE-TASK: PREDICTING FARE")
print("==================================================")
X_reg = df_model.drop(columns=["fare"])
y_reg = df_model["fare"]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

reg_num_features = ("age", "sibsp", "parch")
reg_cat_features = ("sex", "embarked", "pclass", "survived")

reg_preprocessor = ColumnTransformer(transformers=(
    ("num", Pipeline(steps=(
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    )), list(reg_num_features)),
    ("cat", Pipeline(steps=(
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    )), list(reg_cat_features))
))

reg_pipe = Pipeline(steps=(
    ("preprocessor", reg_preprocessor),
    ("regressor", LinearRegression())
))
reg_pipe.fit(X_reg_train, y_reg_train)
y_reg_pred = reg_pipe.predict(X_reg_test)

mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = float(np.sqrt(mean_squared_error(y_reg_test, y_reg_pred)))
r2 = r2_score(y_reg_test, y_reg_pred)
n_test = len(y_reg_test)
p_features = len(X_reg_train.columns)
adj_r2 = 1 - (1 - r2) * (n_test - 1) / (n_test - p_features - 1)

print("Regression Metrics for Fare Prediction:")
print(f"  MAE:          {mae:.2f}")
print(f"  RMSE:         {rmse:.2f}")
print(f"  R^2:          {r2:.4f}")
print(f"  Adjusted R^2: {adj_r2:.4f}")

# Residual plot
residuals = y_reg_test - y_reg_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_reg_pred, residuals, alpha=0.6, color="teal")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot (Checking for Heteroscedasticity)")
plt.tight_layout()
plt.savefig("analytics/plots/09_regression_residuals.png", dpi=300)
plt.close()
print("Saved Residual plot to analytics/plots/09_regression_residuals.png")
print("Residual Analysis: The expanding funnel shape demonstrates heteroscedasticity (error variance grows with higher fares).")

print("\n==================================================")
print("7. SAVING THE COMPLETE PIPELINE & VERIFYING RELOAD")
print("==================================================")
joblib_path = "analytics/best_pipeline.joblib"
joblib.dump(best_rf_pipe, joblib_path)
print(f"Saved complete end-to-end pipeline to {joblib_path}")

# Verification: Reload and predict on raw input
loaded_pipe = joblib.load(joblib_path)
sample_raw = pd.DataFrame([{
    "pclass": 1,
    "sex": "female",
    "age": 28.0,
    "sibsp": 0,
    "parch": 0,
    "fare": 75.0,
    "embarked": "S"
}])
sample_pred = int(loaded_pipe.predict(sample_raw)[0])
sample_probs = loaded_pipe.predict_proba(sample_raw)
sample_prob_surv = float(sample_probs.take(1))

print("\nRaw Test Passenger Prediction:")
print(f"  Input: 28yo female, 1st class, fare £75, embarked Southampton")
print(f"  Prediction: {'Survived' if sample_pred == 1 else 'Deceased'} (Survival Probability: {sample_prob_surv:.2%})")
print("Persistence verified: Pipeline accepts raw input and produces valid predictions.")
