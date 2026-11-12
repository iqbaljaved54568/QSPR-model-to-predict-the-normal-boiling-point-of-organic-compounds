"""
Per-compound SHAP values, training and test sets (Table S9)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# Export per-compound SHAP values for the final 10-descriptor LightGBM model
# Rebuilds the final model exactly as in 4_model_development/05_final_pipeline_10desc.py
# (same files, same StandardScaler, same locked hyperparameters, raw target),
# checks that it reproduces the published numbers, then saves:
#   shap_values_train_optimal.csv   one row per training compound (n = 4,056)
#   shap_values_test_optimal.csv    one row per test compound     (n = 1,015)
#   shap_global_importance_train_vs_test.csv   mean |SHAP| on both sets
# =============================================================================
pass  # (Google Colab: mounted Google Drive here)
pass  # (Google Colab: mounted Google Drive here)

import numpy as np, pandas as pd, shap, lightgbm
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

TRAIN_FILE = "./BP_FINAL_MODEL/descriptors/train_test_split/train_optimal.csv"
TEST_FILE  = "./BP_FINAL_MODEL/descriptors/train_test_split/test_optimal.csv"
OUT_DIR    = "./BP_FINAL_MODEL/figures/optimal_descriptor_pipeline/"
TARGET_COL_INDEX, DESCRIPTOR_START_INDEX = 5, 6

LGBM_PARAMS = {   # LOCKED_PARAMS["LGBM"] from the optimal-descriptor pipeline
    "n_estimators": 1784, "learning_rate": 0.06686808497475842, "num_leaves": 18,
    "max_depth": 6, "min_child_samples": 32, "subsample": 0.9921028774475024,
    "colsample_bytree": 0.7660499104221706, "reg_alpha": 0.06008952920081579,
    "reg_lambda": 0.001212337250189692, "random_state": 42, "n_jobs": -1, "verbosity": -1,
}

train_df, test_df = pd.read_csv(TRAIN_FILE), pd.read_csv(TEST_FILE)
feature_names = list(train_df.columns[DESCRIPTOR_START_INDEX:])
id_cols = list(train_df.columns[:DESCRIPTOR_START_INDEX])          # SMILES, name, Tb, etc.
X_train = train_df.iloc[:, DESCRIPTOR_START_INDEX:].values
y_train = train_df.iloc[:, TARGET_COL_INDEX].values
X_test  = test_df.iloc[:, DESCRIPTOR_START_INDEX:].values
y_test  = test_df.iloc[:, TARGET_COL_INDEX].values

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

model = LGBMRegressor(**LGBM_PARAMS).fit(X_train_sc, y_train)
print(f"Test R2 = {r2_score(y_test, model.predict(X_test_sc)):.4f}   (published: 0.9596)")

explainer = shap.TreeExplainer(model)          # default: tree_path_dependent, no background set
print("feature_perturbation:", explainer.feature_perturbation,
      "| expected value (K):", float(np.ravel(explainer.expected_value)[0]))
sv_train = explainer.shap_values(X_train_sc)
sv_test  = explainer.shap_values(X_test_sc)

# additivity check: base value + sum of SHAP = model prediction
base = float(np.ravel(explainer.expected_value)[0])
print("Max additivity error (K): train %.2e, test %.2e" % (
    np.abs(base + sv_train.sum(1) - model.predict(X_train_sc)).max(),
    np.abs(base + sv_test.sum(1) - model.predict(X_test_sc)).max()))

imp_train = pd.Series(np.abs(sv_train).mean(0), index=feature_names)
imp_test  = pd.Series(np.abs(sv_test).mean(0),  index=feature_names)
glob = pd.DataFrame({"Mean_abs_SHAP_train_K": imp_train, "Mean_abs_SHAP_test_K": imp_test})
glob["Share_train_%"] = 100 * glob["Mean_abs_SHAP_train_K"] / glob["Mean_abs_SHAP_train_K"].sum()
glob["Share_test_%"]  = 100 * glob["Mean_abs_SHAP_test_K"]  / glob["Mean_abs_SHAP_test_K"].sum()
glob = glob.sort_values("Mean_abs_SHAP_train_K", ascending=False)
glob["Rank_train"] = range(1, len(glob) + 1)
glob["Rank_test"]  = glob["Mean_abs_SHAP_test_K"].rank(ascending=False).astype(int)
print(glob.round(3).to_string())
rho, p = spearmanr(glob["Mean_abs_SHAP_train_K"], glob["Mean_abs_SHAP_test_K"])
print(f"\nSpearman rank correlation, train vs test importance: rho = {rho:.3f} (p = {p:.2g})")
print("Published check: rdkit2d_Ipc train mean|SHAP| should be 58.84 K ->",
      round(imp_train["rdkit2d_Ipc"], 2))

def export(df, sv, X_raw, pred, name):
    out = df[id_cols].copy()
    out["Predicted_Tb_K"] = pred
    out["SHAP_base_value_K"] = base
    for k, f in enumerate(feature_names):
        out[f"SHAP_{f}"] = sv[:, k]
    out.to_csv(f"{OUT_DIR}{name}", index=False)
    print("Saved ->", OUT_DIR + name, out.shape)

export(train_df, sv_train, X_train, model.predict(X_train_sc), "shap_values_train_optimal.csv")
export(test_df,  sv_test,  X_test,  model.predict(X_test_sc),  "shap_values_test_optimal.csv")
glob.to_csv(f"{OUT_DIR}shap_global_importance_train_vs_test.csv")
print("shap", shap.__version__, "| lightgbm", lightgbm.__version__)
