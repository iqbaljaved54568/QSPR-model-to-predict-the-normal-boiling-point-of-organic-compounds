"""
NIST/DIPPR source flags, public dataset and final model files

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# Independent-verification package (NIST vs DIPPR source flags)
#
# 1. Flags every compound's boiling-point SOURCE by matching CAS numbers against
#    your raw DIPPR file: if the Tb in the final dataset equals the DIPPR value,
#    the value is DIPPR-licensed; otherwise it is the NIST value.
# 2. Rebuilds the final 10-descriptor LightGBM (same as published) and reports
#    test metrics for ALL, NIST-only and DIPPR-only test compounds.
# 3. Saves the files for Zenodo:
#    - dataset_public_5071.csv   all compounds: Name, CAS, SMILES, Formula, MW,
#                                Source, Split, Tb (blank for DIPPR), 10 descriptors
#    - final_model_lightgbm.txt  trained model (LightGBM native format)
#    - final_model_scaler.json   StandardScaler means/SDs for the 10 descriptors
#    - verify_nist_test.py       script anyone can run to reproduce the NIST metrics
# =============================================================================
pass  # (Google Colab: mounted Google Drive here)
pass  # (Google Colab: mounted Google Drive here)

import json, numpy as np, pandas as pd, lightgbm
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

TRAIN_FILE = "./BP_FINAL_MODEL/descriptors/train_test_split/train_optimal.csv"
TEST_FILE  = "./BP_FINAL_MODEL/descriptors/train_test_split/test_optimal.csv"
DIPPR_FILE = "./BP_FINAL_MODEL/DIPPER.csv"   # your raw DIPPR export
OUT_DIR    = "./BP_FINAL_MODEL/zenodo_release/"
TB_TOL     = 0.05   # K; values closer than this are treated as the same (DIPPR) value

import os; os.makedirs(OUT_DIR, exist_ok=True)
TARGET_COL_INDEX, DESCRIPTOR_START_INDEX = 5, 6
LGBM_PARAMS = {"n_estimators": 1784, "learning_rate": 0.06686808497475842, "num_leaves": 18,
    "max_depth": 6, "min_child_samples": 32, "subsample": 0.9921028774475024,
    "colsample_bytree": 0.7660499104221706, "reg_alpha": 0.06008952920081579,
    "reg_lambda": 0.001212337250189692, "random_state": 42, "n_jobs": -1, "verbosity": -1}

train_df, test_df = pd.read_csv(TRAIN_FILE), pd.read_csv(TEST_FILE)
TB = train_df.columns[TARGET_COL_INDEX]
feats = list(train_df.columns[DESCRIPTOR_START_INDEX:])
print("Metadata columns:", list(train_df.columns[:DESCRIPTOR_START_INDEX]))

# ---- 1. Source flags --------------------------------------------------------
dippr = pd.read_csv(DIPPR_FILE)
print("DIPPR file columns:", list(dippr.columns))
cas_col = next((c for c in dippr.columns if "cas" in c.lower()), None)
tb_col  = next((c for c in dippr.columns if "boil" in c.lower()), None)
if cas_col is None or tb_col is None:
    raise ValueError("Could not find CAS / boiling-point columns in the DIPPR file -- "
                     "set cas_col and tb_col by hand from the column list printed above.")
norm = lambda s: str(s).strip().lstrip("0")
dippr_tb = (dippr.assign(_cas=dippr[cas_col].map(norm))
                 .groupby("_cas")[tb_col].apply(lambda v: list(pd.to_numeric(v, errors="coerce"))))

def flag(row):
    vals = dippr_tb.get(norm(row["CAS No."]))
    if vals is None:
        return "NIST"
    return "DIPPR" if any(abs(row[TB] - v) <= TB_TOL for v in vals if pd.notna(v)) else "NIST (also in DIPPR)"

for df, split in [(train_df, "train"), (test_df, "test")]:
    df["Source"] = df.apply(flag, axis=1)
    df["Split"] = split
full = pd.concat([train_df, test_df], ignore_index=True)
print("\nCompounds by source of the Tb value:")
print(pd.crosstab(full["Source"], full["Split"], margins=True))

# ---- 2. Rebuild final model and evaluate by source ---------------------------
scaler = StandardScaler().fit(train_df[feats].values)
model = LGBMRegressor(**LGBM_PARAMS).fit(scaler.transform(train_df[feats].values), train_df[TB].values)
test_df["Predicted_Tb_K"] = model.predict(scaler.transform(test_df[feats].values))

def metrics(d):
    y, p = d[TB].values, d["Predicted_Tb_K"].values
    return pd.Series({"n": len(d), "R2": r2_score(y, p), "MAE_K": mean_absolute_error(y, p),
                      "RMSE_K": mean_squared_error(y, p) ** 0.5})
is_nist = test_df["Source"].str.startswith("NIST")
res = pd.DataFrame({"All test (published)": metrics(test_df),
                    "NIST-sourced test": metrics(test_df[is_nist]),
                    "DIPPR-sourced test": metrics(test_df[~is_nist])}).T
print("\n", res.round(4).to_string())
res.to_csv(f"{OUT_DIR}test_metrics_by_source.csv")

# ---- 3. Files for the public repository ------------------------------------
meta = list(train_df.columns[:TARGET_COL_INDEX])
pub = full[meta + ["Source", "Split", TB] + feats].copy()
pub.loc[pub["Source"] == "DIPPR", TB] = np.nan          # licence-restricted values withheld
pub = pub.rename(columns={TB: "Tb_K (blank = DIPPR, licence-restricted)"})
pub.to_csv(f"{OUT_DIR}dataset_public_5071.csv", index=False)

model.booster_.save_model(f"{OUT_DIR}final_model_lightgbm.txt")
json.dump({"features": feats, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
           "lightgbm_version": lightgbm.__version__}, open(f"{OUT_DIR}final_model_scaler.json", "w"), indent=1)

verify = '''# Reproduces the NIST-only test metrics from public files only.
import json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
d = pd.read_csv("dataset_public_5071.csv")
s = json.load(open("final_model_scaler.json"))
m = lgb.Booster(model_file="final_model_lightgbm.txt")
t = d[(d["Split"] == "test") & d["Source"].str.startswith("NIST")]
X = (t[s["features"]].values - np.array(s["mean"])) / np.array(s["scale"])
y, p = t.iloc[:, t.columns.get_loc("Split") + 1].values, m.predict(X)
print(f"NIST test: n={len(t)}  R2={r2_score(y, p):.4f}  MAE={mean_absolute_error(y, p):.3f} K  "
      f"RMSE={mean_squared_error(y, p) ** 0.5:.3f} K")
'''
open(f"{OUT_DIR}verify_nist_test.py", "w").write(verify)
print(f"\nSaved release files to {OUT_DIR}")

# ---- 4. Self-test: run the public verification script exactly as a reader would
os.chdir(OUT_DIR); exec(verify)
