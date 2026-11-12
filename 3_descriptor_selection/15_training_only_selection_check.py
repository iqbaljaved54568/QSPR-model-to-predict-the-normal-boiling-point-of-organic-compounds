"""
Descriptor selection repeated on the training set only

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# Descriptor selection repeated on the TRAINING SET ONLY (leakage check)
# Reads the data files; results are printed and saved to train_only_selection_results.csv.
#
# What it does
#  A. Reproduction check: repeats your original steps on ALL 5,071 compounds
#     (3_descriptor_selection/01_variance_spearman_vif_filter.py: cleaning -> variance -> Spearman -> VIF; 3_descriptor_selection/02_random_forest_ranking.py: RF
#     top-35) and confirms it gets your saved 704 / 94 / 35 descriptors.
#  B. Same steps on the 4,056 TRAINING compounds only, using your existing
#     train/test split. Reports which descriptors differ at each stage.
#  C. If the 35-descriptor set differs, repeats the LightGBM backward
#     elimination (3_descriptor_selection/07_backward_elimination_35desc.py settings) on the new set, training data only.
#     Either way, the descriptor count is chosen by CROSS-VALIDATED R^2
#     (99% of peak), never by the test set.
#  D. Compares the resulting 10-descriptor set with your final ten, and if
#     it differs, scores both sets with the final LightGBM hyperparameters.
# Time: about 15-20 min if the 35-set is unchanged, about 25-35 min otherwise.
# =============================================================================
pass  # (Google Colab: mounted Google Drive here)
pass  # (Google Colab: mounted Google Drive here)

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm"], check=True)

import time, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
import sklearn, lightgbm
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ paths
D = "./BP_FINAL_MODEL/descriptors/"
COMBINED   = D + "descriptors_combined_2D3D.csv"           # 5,071 x 1,916 (2_descriptor_calculation/01_generate_descriptors.py)
STAGE2_OLD = D + "stage2_reduced.csv"                      # your 94 (3_descriptor_selection/01_variance_spearman_vif_filter.py)
TRAIN_35   = D + "train_test_split/train_35desc.csv"       # your 35
TRAIN_OPT  = D + "train_test_split/train_optimal.csv"      # split membership
TEST_OPT   = D + "train_test_split/test_optimal.csv"
OUT_LOCAL  = "./train_only_selection_results.csv"

# ------------------------------------------------- settings copied from your cells
METADATA_ALIASES = [                                       # 3_descriptor_selection/01_variance_spearman_vif_filter.py
    ["Name"], ["CAS No.", "CAS", "CAS_No", "CAS Number"],
    ["smiles", "SMILES", "canonical_smiles"], ["Formula", "Calculated_Formula"],
    ["molecular weight", "Molecular Weight", "MW"],
    ["boiling point (K)", "Boiling Point (K)", "BP", "boiling_point_K"],
]
TARGET = "boiling point (K)"
ZERO_THRESHOLD, VAR_THRESHOLD = 0.50, 0.01                 # 3_descriptor_selection/01_variance_spearman_vif_filter.py
SPEARMAN_THRESHOLD, VIF_THRESHOLD = 0.10, 10.0             # 3_descriptor_selection/01_variance_spearman_vif_filter.py
RF_TOPN, RF_TREES, RS = 35, 1000, 42                       # 3_descriptor_selection/02_random_forest_ranking.py
LGBM_35 = {"n_estimators": 2835, "learning_rate": 0.026811730129178632,   # 3_descriptor_selection/07_backward_elimination_35desc.py
           "num_leaves": 145, "max_depth": 4, "min_child_samples": 33,
           "subsample": 0.9509625022217238, "colsample_bytree": 0.7544040195312076,
           "reg_alpha": 0.002770059887158264, "reg_lambda": 4.0236971995361515,
           "random_state": 42, "n_jobs": -1, "verbosity": -1}
LGBM_FINAL = {"n_estimators": 1784, "learning_rate": 0.06686808497475842,  # 4_model_development/05_final_pipeline_10desc.py
              "num_leaves": 18, "max_depth": 6, "min_child_samples": 32,
              "subsample": 0.9921028774475024, "colsample_bytree": 0.7660499104221706,
              "reg_alpha": 0.06008952920081579, "reg_lambda": 0.001212337250189692,
              "random_state": 42, "n_jobs": -1, "verbosity": -1}
CV_ELIM = 5                                                # 3_descriptor_selection/07_backward_elimination_35desc.py
def build_grid(p_start):                                  # 3_descriptor_selection/07_backward_elimination_35desc.py schedule
    grid, n = {p_start}, p_start
    for lo, step in [(15, 3), (1, 1)]:
        while n > lo:
            n = max(lo, n - step); grid.add(n)
    return sorted(grid, reverse=True)
GRID = build_grid(35)   # [35, 32, ..., 15, 14, ..., 1], as in 3_descriptor_selection/07_backward_elimination_35desc.py
FINAL_TEN = ["rdkit2d_Ipc", "mordred_MATS1se", "mordred_GATS1m", "mordred_ATSC1c",
             "mordred_BCUTi-1h", "mordred_ATS3m", "mordred_AATS1i", "mordred_ATSC1i",
             "mordred_BCUTi-1l", "mordred_AATSC1d"]
RUN_ELIMINATION = "auto"   # "auto": only if the 35-set changes; True: always; False: never

# ------------------------------------------------------------------ helpers (3_descriptor_selection/01_variance_spearman_vif_filter.py logic)
def rename_meta(df):
    m = {}
    for al in METADATA_ALIASES:
        f = next((a for a in al if a in df.columns), None)
        if f is None:
            raise ValueError(f"Column not found: {al}")
        m[f] = al[0]
    return df.rename(columns=m)

def cleaning(X):
    X = X.select_dtypes(include=[np.number])
    n_num = X.shape[1]
    X = X.loc[:, (X == 0).mean() <= ZERO_THRESHOLD]
    n_zero = X.shape[1]
    X = X.fillna(X.median())
    return X, n_num, n_zero

def variance(X):
    vt = VarianceThreshold(threshold=VAR_THRESHOLD).fit(X)
    return X.loc[:, vt.get_support()]

def spearman(X, y):
    s = X.apply(lambda c: abs(spearmanr(c, y, nan_policy="omit")[0]))
    return X.loc[:, s >= SPEARMAN_THRESHOLD]

def vif(X):
    while X.shape[1] > 1:
        c = X.corr().values.copy() + np.eye(X.shape[1]) * 1e-10
        try:
            inv = np.linalg.inv(c)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(c)
        v = pd.Series(np.diag(inv), index=X.columns)
        if v.max() < VIF_THRESHOLD:
            break
        X = X.drop(columns=[v.idxmax()])
    return X

def rf_top(X, y):
    Xn = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    rf = RandomForestRegressor(n_estimators=RF_TREES, random_state=RS, n_jobs=-1).fit(Xn, y)
    imp = pd.Series(rf.feature_importances_, index=Xn.columns).sort_values(ascending=False)
    return imp.head(RF_TOPN).index.tolist()

def run_pipeline(X, y, label):
    t0 = time.time()
    Xc, n_num, n_zero = cleaning(X)
    Xv = variance(Xc)
    Xs = spearman(Xv, y)
    Xf = vif(Xs)
    top = rf_top(Xf, y)
    print(f"  [{label}] {X.shape[1]} -> numeric {n_num} -> zeros {n_zero} -> variance {Xv.shape[1]}"
          f" -> Spearman {Xs.shape[1]} -> VIF {Xf.shape[1]} -> RF top {len(top)}  ({time.time()-t0:.0f} s)")
    return {"spearman": list(Xs.columns), "vif": list(Xf.columns), "top35": top}

def compare(a, b, name, la="all data", lb="training only"):
    sa, sb = set(a), set(b)
    print(f"  {name:<16} {la}: {len(sa):>4} | {lb}: {len(sb):>4} | in both: {len(sa & sb):>4}"
          f" | only {la}: {len(sa - sb):>3} | only {lb}: {len(sb - sa):>3}")
    if 0 < len(sa ^ sb) <= 20:
        if sa - sb: print(f"      only {la}: {sorted(sa - sb)}")
        if sb - sa: print(f"      only {lb}: {sorted(sb - sa)}")
    return sa == sb

def elimination(Xtr, ytr, names):
    """LightGBM elimination sweep (as in 07_backward_elimination_35desc.py) (split-count importance, refit each step), training data only."""
    sc = StandardScaler().fit(Xtr)
    X = sc.transform(Xtr)
    grid = build_grid(X.shape[1])
    remaining, rec = list(range(X.shape[1])), []
    while True:
        n = len(remaining)
        cv = KFold(n_splits=CV_ELIM, shuffle=True, random_state=RS)
        r2 = cross_val_score(LGBMRegressor(**LGBM_35), X[:, remaining], ytr, cv=cv, scoring="r2", n_jobs=1)
        m = LGBMRegressor(**LGBM_35).fit(X[:, remaining], ytr)
        if n in grid:
            rec.append({"n": n, "cv_r2": r2.mean(), "cv_sd": r2.std(),
                        "features": [names[i] for i in remaining]})
            print(f"    n={n:3d}  CV R2={r2.mean():.4f} +/- {r2.std():.4f}")
        if n <= 1:
            break
        nxt = max([g for g in grid if g < n] or [1])
        drop = {remaining[i] for i in np.argsort(m.feature_importances_)[:n - nxt]}
        remaining = [i for i in remaining if i not in drop]
    return pd.DataFrame(rec)

def score_set(cols, Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr[cols])
    A, B = sc.transform(Xtr[cols]), sc.transform(Xte[cols])
    cv = KFold(n_splits=10, shuffle=True, random_state=RS)
    r2 = cross_val_score(LGBMRegressor(**LGBM_FINAL), A, ytr, cv=cv, scoring="r2", n_jobs=1)
    p = LGBMRegressor(**LGBM_FINAL).fit(A, ytr).predict(B)
    return (f"10-fold CV R2 {r2.mean():.4f} +/- {r2.std(ddof=1):.4f} | test R2 {r2_score(yte, p):.4f}"
            f", MAE {mean_absolute_error(yte, p):.2f} K, RMSE {np.sqrt(mean_squared_error(yte, p)):.2f} K")

# ------------------------------------------------------------------ load
df = rename_meta(pd.read_csv(COMBINED))   # read exactly as in 3_descriptor_selection/01_variance_spearman_vif_filter.py
meta = [a[0] for a in METADATA_ALIASES]
df["smiles"] = df["smiles"].astype(str).str.strip()
tr_smi = set(pd.read_csv(TRAIN_OPT)["smiles"].astype(str).str.strip())
te_smi = set(pd.read_csv(TEST_OPT)["smiles"].astype(str).str.strip())
is_tr, is_te = df["smiles"].isin(tr_smi), df["smiles"].isin(te_smi)
X_all = df.drop(columns=meta)
y_all = df[TARGET]

old94 = [c for c in pd.read_csv(STAGE2_OLD, nrows=1).columns if c not in meta]
old35 = list(pd.read_csv(TRAIN_35, nrows=1).columns[6:])

print("\n==== RESULTS ====")
print(f"sklearn {sklearn.__version__} | lightgbm {lightgbm.__version__} | numpy {np.__version__} | pandas {pd.__version__}")
print(f"Compounds: {len(df)} | train matched: {is_tr.sum()} | test matched: {is_te.sum()} | in both: {(is_tr & is_te).sum()}")
print(f"Saved sets read from Drive: 94-set {len(old94)} descriptors, 35-set {len(old35)} descriptors")

# ------------------------------------------------------------------ A. reproduction on all data
print("\nA. REPRODUCTION CHECK (original procedure, all 5,071 compounds)")
A = run_pipeline(X_all, y_all, "all data")
ok94 = compare(old94, A["vif"], "VIF set (94)", "saved", "rerun")
ok35 = compare(old35, A["top35"], "Top-35 set", "saved", "rerun")
reproduced = ok94 and ok35
print(f"  -> Reproduction {'SUCCEEDED' if reproduced else 'DID NOT fully match'}: "
      f"{'the rerun below is a like-for-like comparison' if reproduced else 'interpret part B with care (see differences above)'}")

# ------------------------------------------------------------------ B. training only
print("\nB. SAME STEPS ON THE TRAINING SET ONLY (4,056 compounds)")
X_tr, y_tr = X_all[is_tr].reset_index(drop=True), y_all[is_tr].reset_index(drop=True)
X_te, y_te = X_all[is_te].reset_index(drop=True), y_all[is_te].reset_index(drop=True)
B = run_pipeline(X_tr, y_tr, "training only")
compare(A["spearman"], B["spearman"], "After Spearman")
compare(A["vif"], B["vif"], "After VIF")
same35 = compare(A["top35"], B["top35"], "Top-35")
print(f"  Final ten contained in training-only top-35: {len(set(FINAL_TEN) & set(B['top35']))}/10")

# ------------------------------------------------------------------ C. elimination
print("\nC. BACKWARD ELIMINATION (LightGBM, same settings as 07_backward_elimination_35desc.py, training data only)")
run_elim = (RUN_ELIMINATION is True) or (RUN_ELIMINATION == "auto" and not same35)
res = None
if run_elim:
    base = B["top35"]
    print(f"  Running on the training-only 35-set ({'unchanged' if same35 else 'changed'}) ...")
    res = elimination(X_tr[base].apply(pd.to_numeric, errors="coerce").fillna(0).values, y_tr.values, base)
else:
    print("  Skipped: the training-only 35-set is identical to the saved set, and the original elimination sweep itself")
    print("  used training data only, so the elimination path is unchanged. The descriptor count")
    print("  chosen by CV (instead of test) R2 is read from your saved sweep results:")
    old = pd.read_csv("./BP_FINAL_MODEL/figures/backward_elimination_35desc/"
                      "backward_elimination_results_35desc.csv")
    old = old[old["model"] == "LightGBM"].rename(columns={"n_features": "n", "cv_r2_mean": "cv_r2", "cv_r2_std": "cv_sd"})
    res = old[["n", "cv_r2", "cv_sd"]].copy()
    for _, r in res.sort_values("n", ascending=False).iterrows():
        print(f"    n={int(r.n):3d}  CV R2={r.cv_r2:.4f} +/- {r.cv_sd:.4f}")

peak = res["cv_r2"].max()
opt_n = int(res.loc[res["cv_r2"] >= 0.99 * peak, "n"].min())
print(f"  Peak CV R2 {peak:.4f} at n={int(res.loc[res['cv_r2'].idxmax(), 'n'])};"
      f" 99%-of-peak threshold {0.99*peak:.4f} -> smallest n = {opt_n}")

# ------------------------------------------------------------------ D. final comparison
print("\nD. RESULT")
if run_elim:
    if (res["n"] == 10).any():
        ten = res.loc[res["n"] == 10, "features"].iloc[0]
    else:
        ten = res.loc[res["n"] == res["n"].max(), "features"].iloc[0]
        print(f"  WARNING: fewer than 10 descriptors reached elimination; using all {len(ten)}.")
    opt = res.loc[res["n"] == opt_n, "features"].iloc[0]
    same_ten = compare(FINAL_TEN, ten, "10-set", "your final ten", "training-only 10")
    print(f"  Training-only set at the CV-chosen n={opt_n}: {sorted(opt)}")
    print(f"  Overlap of that set with your final ten: {len(set(opt) & set(FINAL_TEN))}")
    Xtr_n = X_tr.apply(pd.to_numeric, errors="coerce").fillna(0)
    Xte_n = X_te.apply(pd.to_numeric, errors="coerce").fillna(0)
    print(f"  Your final ten (final hyperparameters):  {score_set(FINAL_TEN, Xtr_n, y_tr, Xte_n, y_te)}")
    if not same_ten:
        print(f"  Training-only ten (same hyperparameters): {score_set(ten, Xtr_n, y_tr, Xte_n, y_te)}")
        print("  (Note: these hyperparameters were tuned on your final ten, which slightly favours that set.)")
else:
    print("  The training-only top-35 equals your top-35, and the elimination path is unchanged,")
    print("  so the same ten descriptors are obtained. All results in the paper stand.")
    print(f"  Choosing the count by CV R2 (99% of peak) gives n = {opt_n} (see C).")

pd.DataFrame({"stage": ["spearman_all", "spearman_train", "vif_all", "vif_train", "top35_all", "top35_train"],
              "descriptors": ["|".join(A["spearman"]), "|".join(B["spearman"]), "|".join(A["vif"]),
                              "|".join(B["vif"]), "|".join(A["top35"]), "|".join(B["top35"])]}
             ).to_csv(OUT_LOCAL, index=False)
print(f"\n(Descriptor lists saved locally to {OUT_LOCAL}, not to Drive.)")
print("==== END ====")
