# BP-Stratified Train/Test Split to Regenerated Descriptor Sets
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. CONFIGURATION
# =============================================================================


PREV_TRAIN_META = "/content/drive/MyDrive/BP_ND_Analysis/BP_descriptor_outputs/train_test_split/train_63desc.csv"
PREV_TEST_META  = "/content/drive/MyDrive/BP_ND_Analysis/BP_descriptor_outputs/train_test_split/test_63desc.csv"

# --- New, regenerated descriptor files (post second-round VIF / RF) --------
FILE_STAGE2 = "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/stage2_reduced.csv"       # ~94 descriptors
FILE_TOP35  = "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/top35_descriptors.csv"     # ~35 descriptors

OUT_DIR = "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/train_test_split/"
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

#===========================================================================
# Data Structure Summary - Train/Test Split Files
# Verifies column layout (5 metadata + 1 target + N descriptors) and reports
# a summary table across all four split files before model development.
#===========================================================================

import pandas as pd

# ============================================================
# CONFIGURATION - edit paths to your actual four split files
# ============================================================
FILES = {
    "train_97desc": "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/train_test_split/train_97desc.csv",
    "test_97desc":  "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/train_test_split/test_97desc.csv",
    "train_35desc": "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/train_test_split/train_38desc.csv",
    "test_35desc":  "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/train_test_split/test_38desc.csv",
}

N_METADATA_COLS = 5
TARGET_COL_NAME = "boiling point (K)"
TARGET_COL_POSITION = 6   # 1-indexed - the 6th column should be the target


def summarize_file(label, path):
    df = pd.read_csv(path)
    n_rows, n_cols = df.shape

    metadata_cols = df.columns[:N_METADATA_COLS].tolist()
    target_col_actual = df.columns[TARGET_COL_POSITION - 1]
    descriptor_cols = df.columns[TARGET_COL_POSITION:].tolist()
    n_descriptors = n_cols - TARGET_COL_POSITION

    # Sanity check: is the 6th column actually the target?
    target_match = (target_col_actual.strip().lower() == TARGET_COL_NAME.strip().lower())

    return {
        "file": label,
        "total_datapoints": n_rows,
        "metadata_columns": N_METADATA_COLS,
        "target_column": target_col_actual,
        "target_col_ok": target_match,
        "total_descriptors": n_descriptors,
        "total_columns": n_cols,
        "metadata_col_names": metadata_cols,
    }


def main():
    rows = []
    for label, path in FILES.items():
        try:
            rows.append(summarize_file(label, path))
        except FileNotFoundError:
            print(f"⚠️  File not found, skipping: {label} -> {path}")

    summary = pd.DataFrame(rows)

    # Flag any file where the 6th column isn't the expected target name
    bad = summary[~summary["target_col_ok"]]
    if len(bad):
        print("⚠️  WARNING: target column mismatch in the following files "
              "(6th column is not 'boiling point (K)'):")
        print(bad[["file", "target_column"]].to_string(index=False))
        print()

    display_cols = ["file", "total_datapoints", "metadata_columns",
                     "target_column", "total_descriptors", "total_columns"]
    print(summary[display_cols].to_string(index=False))

    print("\nMetadata column names (first file, for reference):")
    if rows:
        print(rows[0]["metadata_col_names"])

    out_path = "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/train_test_split/data_structure_summary.csv"
    summary[display_cols].to_csv(out_path, index=False)
    print(f"\nSaved summary table: {out_path}")

    return summary


if __name__ == "__main__":
    main()

# !pip install optuna

