"""
Applies the same train/test split to the 94- and 35-descriptor sets

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# BP-Stratified Train/Test Split to Regenerated Descriptor Sets
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. CONFIGURATION
# =============================================================================


PREV_TRAIN_META = "./BP_ND_Analysis/BP_descriptor_outputs/train_test_split/train_63desc.csv"
PREV_TEST_META  = "./BP_ND_Analysis/BP_descriptor_outputs/train_test_split/test_63desc.csv"

# --- New, regenerated descriptor files (post second-round VIF / RF) --------
FILE_STAGE2 = "./BP_FINAL_MODEL/descriptors/stage2_reduced.csv"       # ~94 descriptors
FILE_TOP35  = "./BP_FINAL_MODEL/descriptors/top35_descriptors.csv"     # ~35 descriptors

OUT_DIR = "./BP_FINAL_MODEL/descriptors/train_test_split/"
os.makedirs(OUT_DIR, exist_ok=True)

SMILES_COL = "smiles"                 # adjust if different
BP_COL     = "boiling point (K)"      # adjust if different

NON_DESCRIPTOR_COLS_GUESS = {SMILES_COL, BP_COL, "id", "name", "compound", "cid", "index"}

# =============================================================================
# 2. HELPERS
# =============================================================================

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
        for col in df.columns:
            if col.lower().replace(" ", "").replace("_", "") == \
               c.lower().replace(" ", "").replace("_", ""):
                return col
    return None

def count_descriptors(df, smiles_col, bp_col):
    """Dynamically count descriptor columns (everything that isn't an
    obvious metadata column). Used only for naming outputs / printouts,
    so the naming always matches reality even if the count changes again."""
    excluded = set(NON_DESCRIPTOR_COLS_GUESS)
    excluded.add(smiles_col)
    if bp_col:
        excluded.add(bp_col)
    excluded_lower = {c.lower() for c in excluded}
    n = sum(1 for c in df.columns if c.lower() not in excluded_lower)
    return n

# =============================================================================
# 3. LOAD PREVIOUS SPLIT METADATA -> DEFINE TRAIN/TEST SMILES SETS
# =============================================================================

prev_train = pd.read_csv(PREV_TRAIN_META)
prev_test  = pd.read_csv(PREV_TEST_META)

smiles_col_prev_tr = find_col(prev_train, [SMILES_COL])
smiles_col_prev_te = find_col(prev_test, [SMILES_COL])
if smiles_col_prev_tr is None or smiles_col_prev_te is None:
    raise ValueError("Could not find a SMILES column in the previous train/test metadata files.")

train_smiles = set(prev_train[smiles_col_prev_tr].astype(str).str.strip())
test_smiles  = set(prev_test[smiles_col_prev_te].astype(str).str.strip())

overlap = train_smiles & test_smiles
if overlap:
    raise ValueError(f"❌ {len(overlap)} SMILES appear in BOTH previous train and test metadata — "
                      f"cannot proceed until this is resolved.")

print(f"Previous split loaded:")
print(f"  Train SMILES : {len(train_smiles):,}")
print(f"  Test  SMILES : {len(test_smiles):,}")

# =============================================================================
# 4. LOAD NEW DESCRIPTOR FILES
# =============================================================================

df_stage2 = pd.read_csv(FILE_STAGE2)
df_top35  = pd.read_csv(FILE_TOP35)

df_stage2[SMILES_COL] = df_stage2[SMILES_COL].astype(str).str.strip()
df_top35[SMILES_COL]  = df_top35[SMILES_COL].astype(str).str.strip()

bp_col_stage2 = find_col(df_stage2, [BP_COL, "BP", "bp", "boiling_point", "BoilingPoint"])
bp_col_top35  = find_col(df_top35,  [BP_COL, "BP", "bp", "boiling_point", "BoilingPoint"])

n_desc_stage2 = count_descriptors(df_stage2, SMILES_COL, bp_col_stage2)
n_desc_top35  = count_descriptors(df_top35,  SMILES_COL, bp_col_top35)

print(f"\nNew descriptor files:")
print(f"  {FILE_STAGE2}")
print(f"    shape = {df_stage2.shape}  ->  {n_desc_stage2} descriptor columns detected")
print(f"  {FILE_TOP35}")
print(f"    shape = {df_top35.shape}  ->  {n_desc_top35} descriptor columns detected")

# =============================================================================
# 5. ALIGN THE TWO NEW DESCRIPTOR FILES ON SMILES (same logic as before)
# =============================================================================

common_new = set(df_stage2[SMILES_COL]) & set(df_top35[SMILES_COL])
print(f"\nCommon SMILES between stage2 ({n_desc_stage2}-desc) and top35 files: {len(common_new):,}")

df_stage2 = df_stage2[df_stage2[SMILES_COL].isin(common_new)].sort_values(SMILES_COL).reset_index(drop=True)
df_top35  = df_top35[df_top35[SMILES_COL].isin(common_new)].sort_values(SMILES_COL).reset_index(drop=True)

assert (df_stage2[SMILES_COL].values == df_top35[SMILES_COL].values).all(), \
    "❌ Row alignment failed between the two new descriptor files."
print("✅  Row alignment between new descriptor files confirmed")

# =============================================================================
# 6. ASSIGN TRAIN/TEST BASED ON THE PREVIOUS SPLIT (not a new stratified split)
# =============================================================================

is_train = df_stage2[SMILES_COL].isin(train_smiles)
is_test  = df_stage2[SMILES_COL].isin(test_smiles)
is_unseen = ~(is_train | is_test)

n_unseen = is_unseen.sum()
if n_unseen > 0:
    print(f"\n⚠️  {n_unseen} compounds in the new descriptor files were not present in "
          f"the previous train/test metadata (new/regenerated compounds). "
          f"They are EXCLUDED below to preserve the original split's integrity. "
          f"Review them separately if they need to be added to train or test.")

train_stage2 = df_stage2[is_train].reset_index(drop=True)
test_stage2  = df_stage2[is_test].reset_index(drop=True)
train_top35  = df_top35[is_train].reset_index(drop=True)
test_top35   = df_top35[is_test].reset_index(drop=True)

assert (train_stage2[SMILES_COL].values == train_top35[SMILES_COL].values).all()
assert (test_stage2[SMILES_COL].values  == test_top35[SMILES_COL].values).all()

n = len(df_stage2)
print(f"\nTrain : {len(train_stage2):,}  ({len(train_stage2)/n*100:.1f}% of matched compounds)")
print(f"Test  : {len(test_stage2):,}   ({len(test_stage2)/n*100:.1f}% of matched compounds)")

# =============================================================================
# 7. VERIFY BP DISTRIBUTION MATCHES THE ORIGINAL SPLIT'S INTENT
# =============================================================================

if bp_col_stage2:
    print(f"\n  {'Statistic':<12}  {'Train':>10}  {'Test':>10}")
    print(f"  {'─'*36}")
    desc_tr = train_stage2[bp_col_stage2].describe()
    desc_te = test_stage2[bp_col_stage2].describe()
    for stat in ["min", "25%", "50%", "75%", "max", "mean", "std"]:
        print(f"  {stat:<12}  {desc_tr[stat]:>10.2f}  {desc_te[stat]:>10.2f}")

# =============================================================================
# 8. PLOT
# =============================================================================

if bp_col_stage2:
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    fig.suptitle("BP Distribution — Train vs Test (existing split applied to new descriptors)",
                 fontsize=11, fontweight="bold")
    ax.hist(train_stage2[bp_col_stage2], bins=40, alpha=0.6, color="#2C6E8A",
            label=f"Train (n={len(train_stage2):,})", edgecolor="white", linewidth=0.4)
    ax.hist(test_stage2[bp_col_stage2],  bins=40, alpha=0.6, color="#C0392B",
            label=f"Test  (n={len(test_stage2):,})",  edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Boiling Point (K)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.5, linestyle="--"); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR + "fig_bp_split_verification_new_descriptors.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

# =============================================================================
# 9. SAVE — filenames auto-embed the ACTUAL descriptor count for coherence
# =============================================================================

out_train_stage2 = f"{OUT_DIR}train_{n_desc_stage2}desc.csv"
out_test_stage2  = f"{OUT_DIR}test_{n_desc_stage2}desc.csv"
out_train_top35  = f"{OUT_DIR}train_{n_desc_top35}desc.csv"
out_test_top35   = f"{OUT_DIR}test_{n_desc_top35}desc.csv"

train_stage2.to_csv(out_train_stage2, index=False)
test_stage2.to_csv(out_test_stage2,   index=False)
train_top35.to_csv(out_train_top35,   index=False)
test_top35.to_csv(out_test_top35,     index=False)

print(f"\n{'='*60}")
print("  FILES SAVED")
print(f"{'='*60}")
print(f"  {out_train_stage2}  : {train_stage2.shape}")
print(f"  {out_test_stage2}   : {test_stage2.shape}")
print(f"  {out_train_top35}   : {train_top35.shape}")
print(f"  {out_test_top35}    : {test_top35.shape}")
print(f"\n  ✅  Train/test membership matches your ORIGINAL BP-stratified split")
print(f"  ✅  Filenames auto-labelled with actual descriptor counts "
      f"({n_desc_stage2} and {n_desc_top35}) for downstream coherence")

print(f"\n  Update your pipeline:")
print(f"  TRAIN_FILE_{n_desc_stage2} = '{out_train_stage2}'")
print(f"  TEST_FILE_{n_desc_stage2}  = '{out_test_stage2}'")
print(f"  TRAIN_FILE_{n_desc_top35}  = '{out_train_top35}'")
print(f"  TEST_FILE_{n_desc_top35}   = '{out_test_top35}'")
