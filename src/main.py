import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

print("🚀 Program Started...")

# ============================
# LOAD DATA
# ============================

df = pd.read_csv(r"C:\Users\91939\Downloads\final_dataset_full.csv")

print("📊 Dataset Shape:", df.shape)

# ============================
# CLEAN DATA
# ============================

df = df[pd.to_numeric(df["Distance_m"], errors='coerce').notnull()]
df = df.reset_index(drop=True)

for col in df.columns:
    if col != "Scenario":
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

# Remove constant columns
df = df.loc[:, df.nunique() > 1]

print("✅ Data Cleaning Completed")

# ============================
# FEATURE ENGINEERING
# ============================

df["LogDistance"] = np.log10(df["Distance_m"] + 1e-6)
df["LogTotalPower"] = np.log10(df["TotalPower"] + 1e-12)
df["DelaySpreadRatio"] = df["RMSDelay"] / (df["MeanDelay"] + 1e-6)
df["PowerDensity"] = df["TotalPower"] / (df["RMSDelay"] + 1e-6)
df["NormalizedDelay"] = df["MeanDelay"] / (df["Distance_m"] + 1e-6)

if "Scenario" in df.columns:
    df["Scenario"] = df["Scenario"].astype("category").cat.codes

print("✅ Feature Engineering Completed")

# ============================
# FEATURES & TARGET
# ============================

features = [
    "Distance_m",
    "Frequency_GHz",
    "TxHeight",
    "RxHeight",
    "LOS",
    "MeanDelay",
    "RMSDelay",
    "TotalPower",
    "Entropy",
    "LogDistance",
    "LogTotalPower",
    "DelaySpreadRatio",
    "PowerDensity",
    "NormalizedDelay"
]

features = [f for f in features if f in df.columns]

X = df[features]
y = df["PathLoss_dB"]

# ============================
# SCALING
# ============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✅ Data Scaling Completed")

# ============================
# MODELS
# ============================

models = {
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=200,
        max_depth=25,
        n_jobs=-1,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=25,
        n_jobs=-1,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5
    ),

    "Hist Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_depth=8
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    ),

    "Ridge": Ridge(alpha=1.0),

    "Lasso": Lasso(
        alpha=0.01,
        max_iter=10000
    ),

    "SVM": SVR(C=10),

    "KNN": KNeighborsRegressor(
        n_neighbors=7
    )
}

# ============================
# CROSS VALIDATION
# ============================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results_table = []
all_preds_dict = {
    name: np.zeros(len(y))
    for name in models.keys()
}

print("\n🚀 MODEL PERFORMANCE (5-FOLD CV)\n")

for name, model in models.items():

    print(f"Running {name}...")

    rmse_list = []
    r2_list = []

    for train_idx, test_idx in kf.split(X_scaled):

        X_train = X_scaled[train_idx]
        X_test = X_scaled[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        all_preds_dict[name][test_idx] = preds

        rmse = np.sqrt(
            mean_squared_error(y_test, preds)
        )

        r2 = r2_score(y_test, preds)

        rmse_list.append(rmse)
        r2_list.append(r2)

    results_table.append([
        name,
        round(np.mean(rmse_list), 3),
        round(np.mean(r2_list), 3)
    ])

# ============================
# RESULTS TABLE
# ============================

results_df = pd.DataFrame(
    results_table,
    columns=[
        "Model",
        "RMSE (dB)",
        "R² Score"
    ]
)

results_df = results_df.sort_values(
    by="RMSE (dB)"
)

print("\n📊 FINAL RESULTS TABLE\n")
print(results_df)

results_df.to_csv(
    "../results/model_results_table.csv",
    index=False
)

# ============================
# BEST MODEL
# ============================

best_model_name = results_df.iloc[0]["Model"]

print("\n🏆 BEST MODEL:", best_model_name)

all_preds = all_preds_dict[best_model_name]

# ============================
# ACTUAL VS PREDICTED
# ============================

plt.figure(figsize=(8, 6))

plt.scatter(
    y,
    all_preds,
    alpha=0.3
)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    linewidth=2
)

plt.xlabel("Actual Path Loss (dB)")
plt.ylabel("Predicted Path Loss (dB)")
plt.title(
    f"Actual vs Predicted ({best_model_name})"
)

plt.grid(True)

plt.savefig(
    "../images/actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================
# RESIDUAL PLOT
# ============================

residuals = y - all_preds

plt.figure(figsize=(8, 6))

plt.scatter(
    all_preds,
    residuals,
    alpha=0.3
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=2
)

plt.xlabel("Predicted Path Loss (dB)")
plt.ylabel("Residuals (dB)")
plt.title("Residual Analysis")

plt.grid(True)

plt.savefig(
    "../images/residual_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================
# FEATURE IMPORTANCE
# ============================

feat_model = ExtraTreesRegressor(
    n_estimators=200,
    max_depth=25,
    n_jobs=-1,
    random_state=42
)

feat_model.fit(X_scaled, y)

importances = feat_model.feature_importances_

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))

plt.barh(
    range(len(indices)),
    importances[indices]
)

plt.yticks(
    range(len(indices)),
    np.array(X.columns)[indices]
)

plt.xlabel("Importance")
plt.title("Feature Importance")

plt.savefig(
    "../images/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================
# CORRELATION MATRIX
# ============================

corr_df = df[
    features + ["PathLoss_dB"]
].copy()

corr_df = corr_df.loc[
    :,
    corr_df.nunique() > 1
]

corr = corr_df.corr()

plt.figure(figsize=(12, 10))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    square=True
)

plt.title(
    "Feature Correlation Matrix"
)

plt.savefig(
    "../images/correlation_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n✅ Results table saved")
print("✅ Graphs saved successfully")
print("🎉 Project Execution Completed")