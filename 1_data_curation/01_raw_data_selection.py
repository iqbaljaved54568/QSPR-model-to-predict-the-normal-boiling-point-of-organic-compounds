"""
Stratified selection of DIPPR and NIST compounds (10 x 10 boiling point x molecular weight bins)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# ==================================
# Raw Data Selection
# ==================================
import pandas as pd
import numpy as np

# ================= CONFIG =================
INPUT_FILE = "./BP_FINAL_MODEL/DIPPER.csv"
OUTPUT_FILE = "./BP_ND_Analysis/BP_Dipper_1500.csv"

TARGET_SIZE = 1500
BP_COL = "boiling point(K)"      # adjust if needed
MW_COL = "molecular weight"

BP_BINS = 10         # quantiles
MW_BINS = 10
RANDOM_STATE = 42
# =========================================

# ---------- Load data ----------
df = pd.read_csv(INPUT_FILE)

df = df[[BP_COL, MW_COL] + [c for c in df.columns if c not in [BP_COL, MW_COL]]]

df = df.dropna(subset=[BP_COL, MW_COL]).copy()

N = len(df)
print(f"Total compounds available: {N}")

if N <= TARGET_SIZE:
    print("⚠ Dataset already smaller than target size")
    df.to_csv(OUTPUT_FILE, index=False)
    exit()

# ---------- Create quantile bins ----------
df["bp_bin"] = pd.qcut(df[BP_COL], BP_BINS, duplicates="drop")
df["mw_bin"] = pd.qcut(df[MW_COL], MW_BINS, duplicates="drop")

# ---------- Joint stratification ----------
df["stratum"] = df["bp_bin"].astype(str) + "_" + df["mw_bin"].astype(str)

# ---------- Allocate samples proportionally ----------
stratum_counts = df["stratum"].value_counts()
stratum_frac = stratum_counts / len(df)

stratum_target = (stratum_frac * TARGET_SIZE).round().astype(int)

# Fix rounding mismatch
diff = TARGET_SIZE - stratum_target.sum()
if diff != 0:
    stratum_target.iloc[:abs(diff)] += np.sign(diff)

# ---------- Sample ----------
samples = []
for stratum, n in stratum_target.items():
    group = df[df["stratum"] == stratum]
    if n > 0:
        samples.append(group.sample(min(n, len(group)), random_state=RANDOM_STATE))

subset = pd.concat(samples).sample(frac=1, random_state=RANDOM_STATE)

# ---------- Cleanup ----------
subset = subset.drop(columns=["bp_bin", "mw_bin", "stratum"])

subset.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Selected {len(subset)} representative compounds")
print(f"Saved to: {OUTPUT_FILE}")
