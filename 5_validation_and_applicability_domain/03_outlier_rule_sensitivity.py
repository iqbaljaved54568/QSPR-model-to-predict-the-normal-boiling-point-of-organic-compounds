"""
Sensitivity to the boiling-point outlier rule (Tables S17, S18)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# Sensitivity of model statistics to the boiling-point outlier rule
#   1.5 x IQR (published) vs 3 x IQR vs |z| <= 3   (and "no removal")
#
# What it does
#  1. Re-runs curation Steps 0-6 (fast) to recover the pool the IQR rule was
#     applied to (expected 5,082) and the 11 compounds Step 7 removed.
#  2. Calculates the 10 final descriptors for the removed compounds exactly as
#     in the descriptor cell (RDKit Ipc on the H-free molecule; Mordred on the
#     molecule with explicit H), after first CHECKING that this reproduces the
#     stored descriptor values for 100 existing compounds.
#  3. For each outlier rule, with the final 10 descriptors and locked LightGBM
#     hyperparameters held fixed:
#     (A) 10-fold CV on the whole dataset that rule produces;
#     (B) published 1,015-compound test set kept fixed, extra compounds added to
#         the training set -> directly comparable with the published R2 = 0.9596.
#  4. Prints how the published model predicts the 11 removed compounds.
# =============================================================================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit", "mordredcommunity"], check=True)

pass  # (Google Colab: mounted Google Drive here)
pass  # (Google Colab: mounted Google Drive here)

import re, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from mordred import Calculator, descriptors as mord_desc
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
RDLogger.DisableLog("rdApp.*")

BASE = "./BP_FINAL_MODEL/"
RAW_PATH = BASE + "BP_subset_4500.csv"                 # 6,000-row master file (curation input)
MANUAL_PATH = BASE + "manual_exclusion_list.csv"
TRAIN_FILE = BASE + "descriptors/train_test_split/train_optimal.csv"
TEST_FILE = BASE + "descriptors/train_test_split/test_optimal.csv"
OUT_DIR = BASE + "figures/optimal_descriptor_pipeline/"
TB = "boiling point (K)"
LGBM_PARAMS = {"n_estimators": 1784, "learning_rate": 0.06686808497475842, "num_leaves": 18,
    "max_depth": 6, "min_child_samples": 32, "subsample": 0.9921028774475024,
    "colsample_bytree": 0.7660499104221706, "reg_alpha": 0.06008952920081579,
    "reg_lambda": 0.001212337250189692, "random_state": 42, "n_jobs": -1, "verbosity": -1}

def canon(s):
    m = Chem.MolFromSmiles(str(s)) if pd.notna(s) else None
    return Chem.MolToSmiles(m) if m is not None else None

def embeds(smi, attempts=50, seed=42):          # identical to curation Step 5
    m = Chem.MolFromSmiles(smi)
    if m is None: return False
    m = Chem.AddHs(m)
    return AllChem.EmbedMolecule(m, randomSeed=seed, maxAttempts=attempts, useRandomCoords=True) != -1

# ---- 1. Recover the Step-6 pool and the compounds removed by Step 7 ----------
train = pd.read_csv(TRAIN_FILE); test = pd.read_csv(TEST_FILE)
feats = [c for c in train.columns[6:]
         if c.startswith(("rdkit2d_", "mordred_"))]   # the 10 descriptors only
print("VERSION 3 | Descriptors:", feats)
train["Split"], test["Split"] = "train", "test"
final = pd.concat([train, test], ignore_index=True)
final["can"] = final["smiles"].apply(canon)

raw = pd.read_csv(RAW_PATH)
raw["can"] = raw["SMILES"].apply(canon)
manual = pd.read_csv(MANUAL_PATH)
excl = set(manual["Name"].map(raw.set_index("Name")["can"].to_dict()).dropna())
s = raw[~raw["can"].isin(excl)].dropna(subset=["can"]).drop_duplicates("can", keep="first")
allowed = {"C", "H", "N", "O", "S", "P", "Br", "Cl", "I"}
s = s[s["Formula"].apply(lambda f: set(re.findall(r"[A-Z][a-z]?", str(f))).issubset(allowed))]
s = s[s["can"].apply(lambda c: 2 <= Chem.MolFromSmiles(c).GetNumHeavyAtoms() <= 28)]
cand = s[~s["can"].isin(set(final["can"]))].copy()
removed = cand[cand["can"].apply(embeds)].copy()          # Step 5 only needed for these few

pool_tb = pd.concat([final[TB], removed[TB]], ignore_index=True)   # = Step-6 pool
q1, q3 = pool_tb.quantile(0.25), pool_tb.quantile(0.75); iqr = q3 - q1
mu, sd = pool_tb.mean(), pool_tb.std()
print(f"Step-6 pool: {len(pool_tb)} compounds (expected 5,082); removed by Step 7: {len(removed)} (expected 11)")
print(f"Q1 = {q1:.1f} K, Q3 = {q3:.1f} K, IQR = {iqr:.1f} K; mean = {mu:.1f} K, SD = {sd:.1f} K")
if len(removed) != 11:
    print("WARNING: could not recover exactly 11 removed compounds -- send this output before using results.")

# ---- 2. Descriptors for the removed compounds (validated first) --------------
calc = Calculator(mord_desc, ignore_3D=True)
mord_names = [f.replace("mordred_", "") for f in feats if f.startswith("mordred_")]
def calc_desc(smiles_list):
    m2d = [Chem.MolFromSmiles(x) for x in smiles_list]
    md = calc.pandas([Chem.AddHs(m) for m in m2d], quiet=True)[mord_names]
    out = pd.DataFrame({f"mordred_{c}": pd.to_numeric(md[c], errors="coerce").values for c in mord_names})
    out["rdkit2d_Ipc"] = [Descriptors.Ipc(m) for m in m2d]
    return out[feats]

chk = final.sample(100, random_state=0)
rel = (calc_desc(chk["smiles"]).values - chk[feats].values) / np.maximum(np.abs(chk[feats].values), 1e-9)
maxrel = pd.Series(np.nanmax(np.abs(rel), axis=0), index=feats)
print("\nDescriptor reproduction check (max relative difference over 100 compounds):")
print(maxrel.map(lambda v: f"{v:.1e}").to_string())
if (maxrel > 1e-4).any():
    print("WARNING: some descriptors are not reproduced exactly -- send this output before using results.")

removed = removed.reset_index(drop=True)
removed = pd.concat([removed[["Name", "CAS No.", "SMILES", TB]].rename(columns={"SMILES": "smiles"}),
                     calc_desc(removed["SMILES"])], axis=1)
removed["Split"] = "removed"

# ---- 3. Datasets under each rule ---------------------------------------------
pool = pd.concat([final[["Name", "CAS No.", "smiles", TB, "Split"] + feats],
                  removed[["Name", "CAS No.", "smiles", TB, "Split"] + feats]], ignore_index=True)
rules = {
    "1.5 x IQR (published)": (q1 - 1.5 * iqr, q3 + 1.5 * iqr),
    "3 x IQR":               (q1 - 3.0 * iqr, q3 + 3.0 * iqr),
    "|z| <= 3":              (mu - 3 * sd, mu + 3 * sd),
    "No removal":            (-np.inf, np.inf),
}
pub_test = pool[pool["Split"] == "test"]
rows = []
for name, (lo, hi) in rules.items():
    d = pool[(pool[TB] >= lo) & (pool[TB] <= hi)].reset_index(drop=True)
    X, y = d[feats].values, d[TB].values
    # (A) 10-fold CV on the whole dataset this rule produces
    oof, fold = np.zeros(len(d)), []
    for tr, va in KFold(10, shuffle=True, random_state=42).split(X):
        m = make_pipeline(StandardScaler(), LGBMRegressor(**LGBM_PARAMS)).fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
        fold.append((r2_score(y[va], oof[va]), mean_absolute_error(y[va], oof[va]),
                     mean_squared_error(y[va], oof[va]) ** 0.5))
    fold = np.array(fold)
    common = d["Split"].isin(["train", "test"]).values
    # (B) published test set fixed; training = published train + any extra compounds kept
    tr_d = d[d["Split"].isin(["train", "removed"])]
    mB = make_pipeline(StandardScaler(), LGBMRegressor(**LGBM_PARAMS)).fit(tr_d[feats].values, tr_d[TB].values)
    pB = mB.predict(pub_test[feats].values)
    rows.append({"Rule": name, "Bounds (K)": f"{max(lo, 0):.1f}-{hi:.1f}" if np.isfinite(hi) else "none",
                 "N compounds": len(d), "Removed": len(pool) - len(d),
                 "Tb range (K)": f"{y.min():.1f}-{y.max():.1f}",
                 "CV R2": f"{fold[:,0].mean():.4f} ± {fold[:,0].std(ddof=1):.4f}",
                 "CV MAE (K)": f"{fold[:,1].mean():.2f} ± {fold[:,1].std(ddof=1):.2f}",
                 "CV RMSE (K)": f"{fold[:,2].mean():.2f} ± {fold[:,2].std(ddof=1):.2f}",
                 "OOF RMSE, 5,071 common (K)": round(mean_squared_error(y[common], oof[common]) ** 0.5, 2),
                 "Test R2 (fixed 1,015)": round(r2_score(pub_test[TB], pB), 4),
                 "Test MAE (K)": round(mean_absolute_error(pub_test[TB], pB), 3),
                 "Test RMSE (K)": round(mean_squared_error(pub_test[TB], pB) ** 0.5, 3)})
res = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print("\n", res.to_string(index=False))
res.to_csv(OUT_DIR + "outlier_rule_sensitivity.csv", index=False)

# ---- 4. How the published model handles the removed compounds ----------------
pub = make_pipeline(StandardScaler(), LGBMRegressor(**LGBM_PARAMS)).fit(
    pool.loc[pool.Split == "train", feats].values, pool.loc[pool.Split == "train", TB].values)
removed["Predicted_Tb_K"] = pub.predict(removed[feats].values)
removed["Error_K"] = removed["Predicted_Tb_K"] - removed[TB]
print("\nCompounds removed by the 1.5 x IQR rule, predicted by the published model:")
print(removed[["Name", "CAS No.", TB, "Predicted_Tb_K", "Error_K"]].round(1).to_string(index=False))
removed.to_csv(OUT_DIR + "iqr_removed_compounds.csv", index=False)
print("\nSaved -> outlier_rule_sensitivity.csv and iqr_removed_compounds.csv in", OUT_DIR)
