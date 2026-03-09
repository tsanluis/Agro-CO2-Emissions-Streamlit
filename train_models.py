"""
Train all models for the Agri-food CO2 Emission prediction pipeline.
Target: total_emission (kt CO2-eq)
Features: Year + 4 population columns (emission sources excluded to avoid leakage)
All with GridSearchCV hyperparameter tuning and 5-fold cross-validation.
"""

import os, warnings, json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import shap
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

def log(msg):
    print(msg, flush=True)

# ── Paths ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "Agrofood_co2_emission.csv")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 1. Load & Prepare Data ────────────────────────────────────────────
log("Loading data...")
df = pd.read_csv(DATA_PATH)

TARGET = "total_emission"

# Emission source columns are excluded because total_emission = sum(sources)
# Using them would be target leakage (R²=1.0 trivially).
# We keep only Year + population features for a meaningful prediction task.
FEATURE_COLS = [
    "Year",
    "Rural population",
    "Urban population",
    "Total Population - Male",
    "Total Population - Female",
]

log(f"Target: {TARGET}")
log(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

X = df[FEATURE_COLS].values
y = df[TARGET].values

# ── 2. Train / Test Split FIRST (no data leakage) ────────────────────
log("Splitting data 70/30...")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ── 3. Impute (fit on TRAIN only) ────────────────────────────────────
log("Imputing (median, fit on train only)...")
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train_raw)
X_test_imp = imputer.transform(X_test_raw)
X_full_imp = imputer.transform(X)

# ── 4. Scale (fit on TRAIN only) ─────────────────────────────────────
log("Scaling (fit on train only)...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_imp)
X_test_sc = scaler.transform(X_test_imp)
X_full_sc = scaler.transform(X_full_imp)

# ── 5. GridSearchCV Hyperparameter Tuning ─────────────────────────────
log("\n" + "=" * 60)
log("HYPERPARAMETER TUNING (GridSearchCV, 5-fold)")
log("=" * 60)

# Lasso
log("\nTuning Lasso...")
lasso_gs = GridSearchCV(
    Lasso(max_iter=10000),
    {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    cv=5, scoring="r2", n_jobs=1,
)
lasso_gs.fit(X_train_sc, y_train)
best_lasso = lasso_gs.best_estimator_
log(f"  Best alpha={lasso_gs.best_params_['alpha']}  CV R²={lasso_gs.best_score_:.4f}")

# Ridge
log("Tuning Ridge...")
ridge_gs = GridSearchCV(
    Ridge(),
    {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    cv=5, scoring="r2", n_jobs=1,
)
ridge_gs.fit(X_train_sc, y_train)
best_ridge = ridge_gs.best_estimator_
log(f"  Best alpha={ridge_gs.best_params_['alpha']}  CV R²={ridge_gs.best_score_:.4f}")

# CART
log("Tuning CART...")
cart_gs = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    {
        "max_depth": [3, 5, 10, 15, 20, None],
        "min_samples_leaf": [1, 3, 5, 10, 20],
        "min_samples_split": [2, 5, 10, 20],
    },
    cv=5, scoring="r2", n_jobs=1,
)
cart_gs.fit(X_train_imp, y_train)
best_cart = cart_gs.best_estimator_
log(f"  Best params={cart_gs.best_params_}  CV R²={cart_gs.best_score_:.4f}")

# Random Forest
log("Tuning Random Forest...")
rf_gs = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [1, 3, 5],
    },
    cv=5, scoring="r2", n_jobs=1,
)
rf_gs.fit(X_train_imp, y_train)
best_rf = rf_gs.best_estimator_
log(f"  Best params={rf_gs.best_params_}  CV R²={rf_gs.best_score_:.4f}")

# LightGBM
log("Tuning LightGBM...")
lgbm_gs = GridSearchCV(
    lgb.LGBMRegressor(random_state=42, verbosity=-1, n_jobs=-1),
    {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 10, -1],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
    },
    cv=5, scoring="r2", n_jobs=1,
)
lgbm_gs.fit(X_train_imp, y_train)
best_lgbm = lgbm_gs.best_estimator_
log(f"  Best params={lgbm_gs.best_params_}  CV R²={lgbm_gs.best_score_:.4f}")

best_params = {
    "Lasso": lasso_gs.best_params_,
    "Ridge": ridge_gs.best_params_,
    "CART (Decision Tree)": cart_gs.best_params_,
    "Random Forest": rf_gs.best_params_,
    "LightGBM": lgbm_gs.best_params_,
}

# ── 6. Final Models ──────────────────────────────────────────────────
lr_model = LinearRegression()
lr_model.fit(X_train_sc, y_train)

models = {
    "Linear Regression": (lr_model, True),
    "Lasso": (best_lasso, True),
    "Ridge": (best_ridge, True),
    "CART (Decision Tree)": (best_cart, False),
    "Random Forest": (best_rf, False),
    "LightGBM": (best_lgbm, False),
}

# ── 7. Evaluate & Cross-Validate ────────────────────────────────────
log("\n" + "=" * 60)
log("FINAL EVALUATION")
log("=" * 60)

results = []
cv_results_all = {}

for name, (model, use_scaled) in models.items():
    Xte = X_test_sc if use_scaled else X_test_imp
    X_full = X_full_sc if use_scaled else X_full_imp

    y_pred = model.predict(Xte)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    scoring = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]
    cv = cross_validate(model, X_full, y, cv=5, scoring=scoring,
                        return_train_score=False, n_jobs=1)

    cv_r2_mean = cv["test_r2"].mean()
    cv_r2_std = cv["test_r2"].std()
    cv_mae_mean = -cv["test_neg_mean_absolute_error"].mean()
    cv_rmse_mean = -cv["test_neg_root_mean_squared_error"].mean()

    results.append({
        "Model": name,
        "Test MAE": round(mae, 2),
        "Test RMSE": round(rmse, 2),
        "Test R²": round(r2, 4),
        "CV R² (mean±std)": f"{cv_r2_mean:.4f} ± {cv_r2_std:.4f}",
        "CV MAE": round(cv_mae_mean, 2),
        "CV RMSE": round(cv_rmse_mean, 2),
    })

    cv_results_all[name] = {
        "r2_scores": cv["test_r2"].tolist(),
        "mae_scores": (-cv["test_neg_mean_absolute_error"]).tolist(),
        "rmse_scores": (-cv["test_neg_root_mean_squared_error"]).tolist(),
    }

    log(f"\n{name}:")
    if name in best_params:
        log(f"  Best params: {best_params[name]}")
    log(f"  Test  -> MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    log(f"  CV(5) -> R²={cv_r2_mean:.4f}±{cv_r2_std:.4f}  MAE={cv_mae_mean:.2f}")

# ── 7b. MLP Neural Network (PyTorch) ──────────────────────────────────
log("\nTraining MLP Neural Network (PyTorch)...")

class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.net(x)

torch.manual_seed(42)

# Scale target for stable MLP training
y_scaler = StandardScaler()
y_train_sc = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_sc_mlp = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
joblib.dump(y_scaler, os.path.join(MODEL_DIR, "y_scaler.pkl"))

mlp = MLPRegressor(X_train_sc.shape[1])
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

X_tr_t = torch.FloatTensor(X_train_sc)
y_tr_t = torch.FloatTensor(y_train_sc).unsqueeze(1)
X_te_t = torch.FloatTensor(X_test_sc)
y_te_t = torch.FloatTensor(y_test_sc_mlp).unsqueeze(1)

# Simple train/val split for history
n_val = int(0.2 * len(X_tr_t))
X_val_t, y_val_t = X_tr_t[-n_val:], y_tr_t[-n_val:]
X_tr_t2, y_tr_t2 = X_tr_t[:-n_val], y_tr_t[:-n_val]

mlp_history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}
best_val_loss = float("inf")
patience_counter = 0
best_state = None

for epoch in range(200):
    mlp.train()
    optimizer.zero_grad()
    pred = mlp(X_tr_t2)
    loss = loss_fn(pred, y_tr_t2)
    loss.backward()
    optimizer.step()

    mlp.eval()
    with torch.no_grad():
        tr_pred = mlp(X_tr_t2)
        val_pred = mlp(X_val_t)
        tr_loss = loss_fn(tr_pred, y_tr_t2).item()
        vl_loss = loss_fn(val_pred, y_val_t).item()
        tr_mae = (tr_pred - y_tr_t2).abs().mean().item()
        vl_mae = (val_pred - y_val_t).abs().mean().item()

    mlp_history["train_loss"].append(tr_loss)
    mlp_history["val_loss"].append(vl_loss)
    mlp_history["train_mae"].append(tr_mae)
    mlp_history["val_mae"].append(vl_mae)

    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        patience_counter = 0
        best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= 15:
            log(f"  Early stopping at epoch {epoch+1}")
            break

if best_state:
    mlp.load_state_dict(best_state)

mlp.eval()
with torch.no_grad():
    mlp_pred_sc = mlp(X_te_t).numpy().flatten()
    mlp_pred = y_scaler.inverse_transform(mlp_pred_sc.reshape(-1, 1)).flatten()
mlp_mae = mean_absolute_error(y_test, mlp_pred)
mlp_rmse = np.sqrt(mean_squared_error(y_test, mlp_pred))
mlp_r2 = r2_score(y_test, mlp_pred)

results.append({
    "Model": "MLP (Neural Network)",
    "Test MAE": round(mlp_mae, 2),
    "Test RMSE": round(mlp_rmse, 2),
    "Test R²": round(mlp_r2, 4),
    "CV R² (mean±std)": "N/A (single train)",
    "CV MAE": round(mlp_mae, 2),
    "CV RMSE": round(mlp_rmse, 2),
})
log(f"  Test -> MAE={mlp_mae:.2f}  RMSE={mlp_rmse:.2f}  R²={mlp_r2:.4f}")

# Save MLP artifacts
torch.save(mlp.state_dict(), os.path.join(MODEL_DIR, "mlp_model.pt"))
with open(os.path.join(MODEL_DIR, "mlp_history.json"), "w") as f:
    json.dump(mlp_history, f)

results_df = pd.DataFrame(results)
log("\n" + "=" * 80)
log(results_df.to_string(index=False))

# ── 8. SHAP ──────────────────────────────────────────────────────────
log("\nComputing SHAP values (Random Forest)...")
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(X_full_imp), size=min(500, len(X_full_imp)), replace=False)
X_shap_sample = X_full_imp[sample_idx]

explainer_rf = shap.TreeExplainer(best_rf)
shap_values_rf = explainer_rf.shap_values(X_shap_sample)

log("Computing SHAP values (LightGBM)...")
explainer_lgbm = shap.TreeExplainer(best_lgbm)
shap_values_lgbm = explainer_lgbm.shap_values(X_shap_sample)

# ── 9. Save ──────────────────────────────────────────────────────────
log("\nSaving artifacts...")
joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "feature_names.pkl"))
joblib.dump(results_df, os.path.join(MODEL_DIR, "model_results.pkl"))

model_filenames = {}
for name, (model, _) in models.items():
    safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    fname = f"{safe}.pkl"
    joblib.dump(model, os.path.join(MODEL_DIR, fname))
    model_filenames[name] = fname

model_filenames["MLP (Neural Network)"] = "mlp_model.pt"
joblib.dump(model_filenames, os.path.join(MODEL_DIR, "model_filenames.pkl"))

# Save CART tree visualization
from sklearn.tree import export_text
tree_text = export_text(best_cart, feature_names=FEATURE_COLS, max_depth=4)
with open(os.path.join(MODEL_DIR, "cart_tree_text.txt"), "w") as f:
    f.write(tree_text)

params_ser = {}
for k, v in best_params.items():
    params_ser[k] = {
        pk: (int(pv) if isinstance(pv, np.integer) else
             float(pv) if isinstance(pv, np.floating) else pv)
        for pk, pv in v.items()
    }
with open(os.path.join(MODEL_DIR, "best_params.json"), "w") as f:
    json.dump(params_ser, f, indent=2)

with open(os.path.join(MODEL_DIR, "cv_results.json"), "w") as f:
    json.dump(cv_results_all, f)

joblib.dump(shap_values_rf, os.path.join(MODEL_DIR, "shap_values_rf.pkl"))
joblib.dump(shap_values_lgbm, os.path.join(MODEL_DIR, "shap_values_lgbm.pkl"))
joblib.dump(X_shap_sample, os.path.join(MODEL_DIR, "X_shap_sample.pkl"))
joblib.dump(explainer_rf, os.path.join(MODEL_DIR, "shap_explainer_rf.pkl"))
joblib.dump(explainer_lgbm, os.path.join(MODEL_DIR, "shap_explainer_lgbm.pkl"))

with open(os.path.join(MODEL_DIR, "split_info.json"), "w") as f:
    json.dump({"test_size": 0.3, "random_state": 42,
               "n_train": len(X_train_sc), "n_test": len(X_test_sc),
               "n_features": len(FEATURE_COLS)}, f)

results_df.to_csv(os.path.join(MODEL_DIR, "model_results.csv"), index=False)

log("\nDone! All artifacts saved.")
