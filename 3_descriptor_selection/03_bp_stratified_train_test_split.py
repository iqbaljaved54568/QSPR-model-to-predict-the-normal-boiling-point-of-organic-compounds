"""
Boiling-point-stratified 80/20 train/test split (10 bins, random_state = 42)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# BP-Stratified Train/Test Split
# Splits both 63-descriptor and 35-descriptor files using identical indices
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

FILE_63   = "./BP_ND_Analysis/BP_descriptor_outputs/final_62desc_aligned.csv"
FILE_35   = "./BP_ND_Analysis/3D/Top35descriptor_Final_clean.csv"

OUT_DIR   = "./BP_ND_Analysis/BP_descriptor_outputs/train_test_split/"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_TRAIN_63 = OUT_DIR + "train_63desc.csv"
OUT_TEST_63  = OUT_DIR + "test_63desc.csv"
OUT_TRAIN_35 = OUT_DIR + "train_35desc.csv"
OUT_TEST_35  = OUT_DIR + "test_35desc.csv"

SMILES_COL   = "smiles"   # adjust if different
BP_COL       = "boiling point (K)"   # adjust if different
TEST_SIZE    = 0.2
N_BINS       = 10
RANDOM_STATE = 42

# =============================================================================
# 2. LOAD
# =============================================================================

df_63 = pd.read_csv(FILE_63)
df_35 = pd.read_csv(FILE_35)

print(f"File 63 desc : {df_63.shape}")
print(f"File 35 desc : {df_35.shape}")

# Auto-detect BP column if name differs
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
        for col in df.columns:
            if col.lower().replace(" ","").replace("_","") == \
               c.lower().replace(" ","").replace("_",""):
                return col
    return None

bp_col_63 = find_col(df_63, [BP_COL, "BP", "bp", "boiling_point", "BoilingPoint"])
bp_col_35 = find_col(df_35, [BP_COL, "BP", "bp", "boiling_point", "BoilingPoint"])

if bp_col_63 is None:
    raise ValueError(f"BP column not found in 63-desc file. Columns: {list(df_63.columns[:10])}")
if bp_col_35 is None:
    raise ValueError(f"BP column not found in 35-desc file. Columns: {list(df_35.columns[:10])}")

print(f"\nBP column (63-desc): '{bp_col_63}'")
print(f"BP column (35-desc): '{bp_col_35}'")

# =============================================================================
# 3. ALIGN BOTH FILES ON SMILES
# =============================================================================

df_63[SMILES_COL] = df_63[SMILES_COL].astype(str).str.strip()
df_35[SMILES_COL] = df_35[SMILES_COL].astype(str).str.strip()

common = set(df_63[SMILES_COL]) & set(df_35[SMILES_COL])
print(f"\nSMILES in 63-desc file : {len(df_63):,}")
print(f"SMILES in 35-desc file : {len(df_35):,}")
print(f"Common SMILES          : {len(common):,}")

df_63 = df_63[df_63[SMILES_COL].isin(common)].sort_values(SMILES_COL).reset_index(drop=True)
df_35 = df_35[df_35[SMILES_COL].isin(common)].sort_values(SMILES_COL).reset_index(drop=True)

assert (df_63[SMILES_COL].values == df_35[SMILES_COL].values).all(), \
    "❌ Row alignment failed — SMILES do not match after sorting"
print("✅  Row alignment confirmed")

# =============================================================================
# 4. CREATE BP BINS FOR STRATIFICATION
# =============================================================================

bp_values = df_63[bp_col_63].values
n         = len(bp_values)

print(f"\nBP range: {bp_values.min():.1f} – {bp_values.max():.1f} K")
print(f"Total compounds: {n:,}")

# Use pd.cut and extract codes directly — avoids all .cat attribute errors
def make_bins(bp_values, n_bins):
    labels     = list(range(n_bins))
    cut_result = pd.cut(bp_values, bins=n_bins, labels=labels)
    codes      = np.array(cut_result, dtype=int)          # integer bin index per compound
    boundaries = np.linspace(bp_values.min(),
                              bp_values.max(), n_bins + 1) # bin edge values
    return codes, boundaries

codes, boundaries = make_bins(bp_values, N_BINS)

# Print distribution
print(f"\nBP bin distribution ({N_BINS} bins):")
for b in range(N_BINS):
    mask  = codes == b
    lo    = boundaries[b]
    hi    = boundaries[b + 1]
    count = mask.sum()
    print(f"  Bin {b:2d}  [{lo:6.1f} – {hi:6.1f} K]  →  {count:4d} compounds")

# If any bin has < 2 samples, reduce bin count
min_count = min((codes == b).sum() for b in range(N_BINS))
while min_count < 2 and N_BINS > 2:
    N_BINS -= 1
    codes, boundaries = make_bins(bp_values, N_BINS)
    min_count = min((codes == b).sum() for b in range(N_BINS))
    print(f"  ⚠️  Reduced to {N_BINS} bins (min bin count was too small)")

# =============================================================================
# 5. STRATIFIED SPLIT
# =============================================================================

indices = np.arange(n)

train_idx, test_idx = train_test_split(
    indices,
    test_size    = TEST_SIZE,
    stratify     = codes,
    random_state = RANDOM_STATE
)

train_idx = np.sort(train_idx)
test_idx  = np.sort(test_idx)

print(f"\nTrain : {len(train_idx):,}  ({len(train_idx)/n*100:.1f}%)")
print(f"Test  : {len(test_idx):,}   ({len(test_idx)/n*100:.1f}%)")

# =============================================================================
# 6. APPLY SAME INDICES TO BOTH FILES
# =============================================================================

train_63 = df_63.iloc[train_idx].reset_index(drop=True)
test_63  = df_63.iloc[test_idx].reset_index(drop=True)
train_35 = df_35.iloc[train_idx].reset_index(drop=True)
test_35  = df_35.iloc[test_idx].reset_index(drop=True)

# =============================================================================
# 7. VERIFY BP DISTRIBUTION
# =============================================================================

print(f"\n  {'Statistic':<12}  {'Train':>10}  {'Test':>10}")
print(f"  {'─'*36}")
desc_tr = train_63[bp_col_63].describe()
desc_te = test_63[bp_col_63].describe()
for stat in ["min", "25%", "50%", "75%", "max", "mean", "std"]:
    print(f"  {stat:<12}  {desc_tr[stat]:>10.2f}  {desc_te[stat]:>10.2f}")

# =============================================================================
# 8. PLOT
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("BP-Stratified Train/Test Split — Verification",
             fontsize=13, fontweight="bold")

# Histogram
ax = axes[0]
ax.hist(train_63[bp_col_63], bins=40, alpha=0.6, color="#2C6E8A",
        label=f"Train (n={len(train_63):,})", edgecolor="white", linewidth=0.4)
ax.hist(test_63[bp_col_63],  bins=40, alpha=0.6, color="#C0392B",
        label=f"Test  (n={len(test_63):,})",  edgecolor="white", linewidth=0.4)
ax.set_xlabel("Boiling Point (K)", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.set_title("(A)  BP Histogram — Train vs Test", loc="left",
             fontsize=10, fontstyle="italic")
ax.legend(fontsize=9)
ax.grid(True, linewidth=0.5, linestyle="--"); ax.set_axisbelow(True)

# Per-bin bar chart
ax2 = axes[1]
train_bin_counts = np.array([(codes[train_idx] == b).sum() for b in range(N_BINS)])
test_bin_counts  = np.array([(codes[test_idx]  == b).sum() for b in range(N_BINS)])
x = np.arange(N_BINS)
w = 0.4
ax2.bar(x - w/2, train_bin_counts, w, color="#2C6E8A", alpha=0.8,
        label="Train", edgecolor="white")
ax2.bar(x + w/2, test_bin_counts,  w, color="#C0392B", alpha=0.8,
        label="Test",  edgecolor="white")
ax2.set_xlabel("BP Bin Index", fontsize=10)
ax2.set_ylabel("Count", fontsize=10)
ax2.set_title("(B)  Compounds per BP Bin", loc="left",
              fontsize=10, fontstyle="italic")
ax2.set_xticks(x)
ax2.legend(fontsize=9)
ax2.grid(True, linewidth=0.5, linestyle="--", axis="y"); ax2.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUT_DIR + "fig_bp_split_verification.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.show()

# =============================================================================
# 9. SAVE
# =============================================================================

train_63.to_csv(OUT_TRAIN_63, index=False)
test_63.to_csv(OUT_TEST_63,   index=False)
train_35.to_csv(OUT_TRAIN_35, index=False)
test_35.to_csv(OUT_TEST_35,   index=False)

print(f"\n{'='*60}")
print("  FILES SAVED")
print(f"{'='*60}")
print(f"  train_63desc.csv : {train_63.shape}")
print(f"  test_63desc.csv  : {test_63.shape}")
print(f"  train_35desc.csv : {train_35.shape}")
print(f"  test_35desc.csv  : {test_35.shape}")
print(f"\n  ✅  Identical indices applied to both descriptor sets")
print(f"\n  Update your pipeline:")
print(f"  TRAIN_FILE = '{OUT_TRAIN_63}'")
print(f"  TEST_FILE  = '{OUT_TEST_63}'")
