"""
Robustness to the selection metric and tree algorithm (Table S13)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# Robustness of the 10-descriptor selection to (a) the selection metric and
# (b) the tree algorithm.
#
# 1. Re-runs the recursive backward elimination from the 35-descriptor set for
#    LightGBM, XGBoost and Random Forest, exactly as in the original sweep (same
#    hyperparameters, importance ranking, schedule, 5-fold CV), but records CV R2,
#    CV MAE and CV RMSE at every step. Checks the descriptor lists against the saved
#    sweep (backward_elimination_features_by_step_35desc.csv).
# 2. Metric: optimal n for LightGBM under R2- and MAE-based rules (training CV only).
# 3. Algorithm: overlap between each model's 10-descriptor set and the final ten,
#    and a cross-evaluation: every algorithm trained on every model's 10-set
#    (10-fold CV R2/MAE on training data + test R2/MAE).
# Outputs -> BP_FINAL_MODEL/figures/backward_elimination_35desc/
# =============================================================================
pass  # (Google Colab: mounted Google Drive here)
pass  # (Google Colab: mounted Google Drive here)
import numpy as np, pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error

S = "./BP_FINAL_MODEL/descriptors/train_test_split/"
OUT = "./BP_FINAL_MODEL/figures/backward_elimination_35desc/"
tr = pd.read_csv(S + "train_35desc.csv"); te = pd.read_csv(S + "test_35desc.csv")
feats = list(tr.columns[6:]); TB = tr.columns[5]
X, y, Xt, yt = tr[feats].values, tr[TB].values, te[feats].values, te[TB].values
final10 = [c for c in pd.read_csv(S + "train_optimal.csv", nrows=0).columns if c.startswith(("rdkit2d_", "mordred_"))]
print(f"{len(feats)} starting descriptors; final ten: {final10}")

BUILD = {   # hyperparameters exactly as in the original sweep (35-descriptor tuned)
 "LightGBM": lambda: LGBMRegressor(n_estimators=2835, learning_rate=0.026811730129178632, num_leaves=145,
     max_depth=4, min_child_samples=33, subsample=0.9509625022217238, colsample_bytree=0.7544040195312076,
     reg_alpha=0.002770059887158264, reg_lambda=4.0236971995361515, random_state=42, n_jobs=-1, verbosity=-1),
 "XGBoost": lambda: XGBRegressor(n_estimators=659, max_depth=4, learning_rate=0.01895242515278871,
     subsample=0.8310238495458468, colsample_bytree=0.7698722276112926, reg_alpha=0.0035336141962440154,
     reg_lambda=0.003230321800809352, min_child_weight=9, gamma=0.008093148207921756,
     random_state=42, n_jobs=-1, verbosity=0, tree_method="hist"),
 "RandomForest": lambda: RandomForestRegressor(n_estimators=871, max_depth=28, min_samples_split=2,
     min_samples_leaf=1, max_features=0.5, random_state=42, n_jobs=-1, bootstrap=True)}
GRID = sorted({35, 32, 29, 26, 23, 20, 17, 15} | set(range(1, 15)), reverse=True)
cv5 = KFold(5, shuffle=True, random_state=42)

# ---- 1. elimination with R2 + MAE + RMSE ----------------------------------------
saved = pd.read_csv(OUT + "backward_elimination_features_by_step_35desc.csv")
recs = []
for name, build in BUILD.items():
    rem = list(range(len(feats)))
    while True:
        n = len(rem)
        cvr = cross_validate(build(), X[:, rem], y, cv=cv5,
                             scoring=("r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"))
        m = build().fit(X[:, rem], y)
        kept = [feats[i] for i in rem]
        s = saved[(saved["model"] == name) & (saved["n_features"] == n)]
        same = (set(s["features_kept"].iloc[0].split("|")) == set(kept)) if len(s) else None
        recs.append({"model": name, "n": n, "cv_r2": cvr["test_r2"].mean(), "cv_r2_sd": cvr["test_r2"].std(ddof=1),
                     "cv_mae": -cvr["test_neg_mean_absolute_error"].mean(),
                     "cv_mae_sd": cvr["test_neg_mean_absolute_error"].std(ddof=1),
                     "cv_rmse": -cvr["test_neg_root_mean_squared_error"].mean(),
                     "matches_saved_sweep": same, "features": "|".join(kept)})
        print(f"  {name:12s} n={n:2d}  CV R2={recs[-1]['cv_r2']:.4f}  CV MAE={recs[-1]['cv_mae']:.2f} K  "
              f"same set as saved sweep: {same}")
        if n <= 1: break
        nxt = max(g for g in GRID if g < n)
        drop = np.argsort(m.feature_importances_)[: n - nxt]
        rem = [r for k, r in enumerate(rem) if k not in set(drop)]
E = pd.DataFrame(recs); E.to_csv(OUT + "elimination_r2_mae_all_tree_models.csv", index=False)

# ---- 2. metric robustness: optimal n under different rules (CV, training only) ---
def rules(d):
    d = d.sort_values("n"); best_r2 = d.loc[d.cv_r2.idxmax()]; best_mae = d.loc[d.cv_mae.idxmin()]
    out = {"n at max CV R2": int(best_r2.n), "n at min CV MAE": int(best_mae.n),
           "R2 >= 99% of max": int(d[d.cv_r2 >= 0.99 * best_r2.cv_r2].n.min()),
           "MAE <= 1.05 x min": int(d[d.cv_mae <= 1.05 * best_mae.cv_mae].n.min()),
           "MAE <= 1.10 x min": int(d[d.cv_mae <= 1.10 * best_mae.cv_mae].n.min()),
           "MAE within 1 SE of min": int(d[d.cv_mae <= best_mae.cv_mae + best_mae.cv_mae_sd / np.sqrt(5)].n.min())}
    return out
R = pd.DataFrame({m: rules(E[E.model == m]) for m in BUILD}).T
print("\nOptimal descriptor count under different rules (training-set 5-fold CV):\n", R.to_string())
R.to_csv(OUT + "optimal_n_by_rule.csv")

# ---- 3. algorithm robustness ------------------------------------------------------
sets = {"Final ten (LightGBM)": final10}
for m in ["XGBoost", "RandomForest"]:
    sets[f"{m} n=10"] = E[(E.model == m) & (E.n == 10)].features.iloc[0].split("|")
print("\nOverlap with the final ten:")
for k, v in sets.items():
    common = set(v) & set(final10)
    print(f"  {k:22s}: {len(common)}/10 shared; not in final ten: {sorted(set(v) - set(final10))}")
freq = pd.Series([f for v in sets.values() for f in v]).value_counts()
print("Descriptors selected by all three models:", sorted(freq[freq == 3].index))

cv10 = KFold(10, shuffle=True, random_state=42); rows = []
for alg, build in BUILD.items():
    for k, v in sets.items():
        idx = [feats.index(f) for f in v]
        cvr = cross_validate(build(), X[:, idx], y, cv=cv10, scoring=("r2", "neg_mean_absolute_error"))
        p = build().fit(X[:, idx], y).predict(Xt[:, idx])
        rows.append({"Algorithm": alg, "Descriptor set": k,
                     "CV R2": f"{cvr['test_r2'].mean():.4f} ± {cvr['test_r2'].std(ddof=1):.4f}",
                     "CV MAE (K)": f"{-cvr['test_neg_mean_absolute_error'].mean():.2f}",
                     "Test R2": round(r2_score(yt, p), 4), "Test MAE (K)": round(mean_absolute_error(yt, p), 2)})
C = pd.DataFrame(rows); print("\nCross-evaluation (10-fold CV on training set; test set):\n", C.to_string(index=False))
C.to_csv(OUT + "cross_evaluation_10desc_sets.csv", index=False)
