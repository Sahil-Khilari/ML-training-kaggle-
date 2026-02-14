import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -----------------------------
# Argument Parser
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--target", type=str, required=True)
parser.add_argument("--kfold", type=int, default=5)
args = parser.parse_args()

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(args.data)

# Drop leakage columns if present
drop_cols = ["instant", "dteday", "casual", "registered"]
for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

X = df.drop(columns=[args.target])
y = df[args.target]

# -----------------------------
# KFold Setup
# -----------------------------
kf = KFold(n_splits=args.kfold, shuffle=True, random_state=42)

results = []

# -----------------------------
# 1️⃣ Bagging → Random Forest
# -----------------------------
rf = RandomForestRegressor(
    n_estimators=200,       # hyperparameter 1
    max_depth=15,           # hyperparameter 2
    random_state=42,
    n_jobs=-1
)

rmse_scores = -cross_val_score(rf, X, y, cv=kf,
                               scoring="neg_root_mean_squared_error")
mae_scores = -cross_val_score(rf, X, y, cv=kf,
                              scoring="neg_mean_absolute_error")

results.append([
    "RandomForest (Bagging)",
    rmse_scores.mean(), rmse_scores.std(),
    mae_scores.mean(), mae_scores.std()
])

# -----------------------------
# 2️⃣ Subagging
# -----------------------------
base_tree = DecisionTreeRegressor(max_depth=15, random_state=42)

subag = BaggingRegressor(
    estimator=base_tree,
    n_estimators=200,      # hyperparameter 1
    max_samples=0.6,       # hyperparameter 2 (<1 = subagging)
    random_state=42,
    n_jobs=-1
)

rmse_scores = -cross_val_score(subag, X, y, cv=kf,
                               scoring="neg_root_mean_squared_error")
mae_scores = -cross_val_score(subag, X, y, cv=kf,
                              scoring="neg_mean_absolute_error")

results.append([
    "Subagging",
    rmse_scores.mean(), rmse_scores.std(),
    mae_scores.mean(), mae_scores.std()
])

# -----------------------------
# 3️⃣ Boosting
# -----------------------------
gbr = GradientBoostingRegressor(
    n_estimators=200,     # hyperparameter 1
    learning_rate=0.05,   # hyperparameter 2
    max_depth=3,
    random_state=42
)

rmse_scores = -cross_val_score(gbr, X, y, cv=kf,
                               scoring="neg_root_mean_squared_error")
mae_scores = -cross_val_score(gbr, X, y, cv=kf,
                              scoring="neg_mean_absolute_error")

results.append([
    "GradientBoosting",
    rmse_scores.mean(), rmse_scores.std(),
    mae_scores.mean(), mae_scores.std()
])

# -----------------------------
# Save CV Results
# -----------------------------
results_df = pd.DataFrame(results,
                          columns=["Model",
                                   "RMSE_mean", "RMSE_std",
                                   "MAE_mean", "MAE_std"])

results_df.to_csv("cv_regression_results.csv", index=False)

print("\nCross-Validation Results:")
print(results_df)

# -----------------------------
# Train Best Model (Boosting usually best)
# -----------------------------
gbr.fit(X, y)
preds = gbr.predict(X)

final_df = pd.DataFrame({
    "ActualCnt": y,
    "PredictedCnt": preds
})

final_df.to_csv("final_predictions.csv", index=False)

# -----------------------------
# Feature Importance
# -----------------------------
importances = gbr.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 8 Important Features:")
print(feature_importance_df.head(8))
