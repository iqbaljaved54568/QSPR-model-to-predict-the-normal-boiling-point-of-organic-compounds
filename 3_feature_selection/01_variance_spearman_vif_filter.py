# Descriptor Reduction Pipeline - Part 1 of 2
# Order: (i) Low-variance removal -> (ii) Spearman relevance filter ->
#        (iii) VIF multicollinearity filter
# -> SAVES stage2_reduced.csv
#===========================================================================

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FILE = "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/descriptors_combined_2D3D.csv"

VARIANCE_OUT = "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/stage_variance_filtered.csv"
SPEARMAN_OUT = "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/stage_spearman_filtered.csv"
VIF_OUT      = "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/stage2_reduced.csv"

# Metadata columns, BY NAME (edit if your column names differ).
METADATA_ALIASES = [
    ["Name"],
    ["CAS No.", "CAS", "CAS_No", "CAS Number"],
    ["smiles", "SMILES", "canonical_smiles"],
    ["Formula", "Calculated_Formula"],
    ["molecular weight", "Molecular Weight", "MW"],
    ["boiling point (K)", "Boiling Point (K)", "BP", "boiling_point_K"],
]
TARGET_COL = "boiling point (K)"  # canonical name (see METADATA_ALIASES above)

ZERO_THRESHOLD = 0.50       # pre-cleaning: drop descriptors with >50% zeros
VAR_THRESHOLD  = 0.01       # (i) low-variance cutoff
SPEARMAN_THRESHOLD = 0.10   # (ii) minimum |correlation to target| to keep
VIF_THRESHOLD  = 10.0       # (iii) iterative VIF cutoff


def resolve_metadata_columns(df):
    rename_map = {}
    not_found = []
    for aliases in METADATA_ALIASES:
        canonical = aliases[0]
        found = next((a for a in aliases if a in df.columns), None)
        if found is None:
            not_found.append(aliases)
        else:
            rename_map[found] = canonical
    if not_found:
        raise ValueError(
            f"Could not find a match for these metadata fields (tried these "
            f"name variants for each): {not_found}. "
            f"Available columns (first 15): {list(df.columns)[:15]}"
        )
    return rename_map


def load_and_split(path):
    df = pd.read_csv(path)
    rename_map = resolve_metadata_columns(df)
    df = df.rename(columns=rename_map)
    metadata_cols = [aliases[0] for aliases in METADATA_ALIASES]
    metadata = df[metadata_cols].copy()
    target = df[TARGET_COL].copy()
    descriptors = df.drop(columns=metadata_cols).copy()
    return metadata, target, descriptors


def basic_cleaning(descriptors):
    """Numeric-only + drop mostly-zero columns. Pre-processing, not one of
    the three selection steps below, but required before variance/VIF will
    work correctly."""
    print(f"[Cleaning] Initial descriptors: {descriptors.shape[1]}")
    descriptors = descriptors.select_dtypes(include=[np.number])
    print(f"[Cleaning] After non-numeric removal: {descriptors.shape[1]}")

    zero_fraction = (descriptors == 0).mean()
    descriptors = descriptors.loc[:, zero_fraction <= ZERO_THRESHOLD]
    print(f"[Cleaning] After >{ZERO_THRESHOLD:.0%} zero filter: {descriptors.shape[1]}")

    descriptors = descriptors.fillna(descriptors.median())
    return descriptors


# ============================================================
# (i) LOW-VARIANCE REMOVAL
# ============================================================
def step_variance_filter(descriptors):
    vt = VarianceThreshold(threshold=VAR_THRESHOLD)
    vt_array = vt.fit_transform(descriptors)
    descriptors = pd.DataFrame(
        vt_array, columns=descriptors.columns[vt.get_support()], index=descriptors.index
    )
    print(f"[Step i - Variance] After low-variance filter (>{VAR_THRESHOLD}): {descriptors.shape[1]}")
    return descriptors


# ============================================================
# (ii) SPEARMAN RELEVANCE FILTER
# ============================================================
def step_spearman_filter(descriptors, target):
    spearman_scores = descriptors.apply(
        lambda c: abs(spearmanr(c, target, nan_policy="omit")[0])
    )
    descriptors = descriptors.loc[:, spearman_scores >= SPEARMAN_THRESHOLD]
    print(f"[Step ii - Spearman] After |corr| >= {SPEARMAN_THRESHOLD} filter: {descriptors.shape[1]}")
    return descriptors


# ============================================================
# (iii) VIF MULTICOLLINEARITY FILTER (fast, correlation-matrix based)
# ============================================================
def fast_vif(descriptors):
    """VIF via correlation-matrix inversion: VIF_i = diag(inv(corr_matrix))_i.
    Equivalent to the standard OLS-based VIF for standardized features, and
    dramatically faster than fitting one regression per feature per iteration."""
    corr = descriptors.corr().values.copy()
    corr += np.eye(corr.shape[0]) * 1e-10  # ridge term for numerical stability
    try:
        inv_corr = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        inv_corr = np.linalg.pinv(corr)
    vif = np.diag(inv_corr)
    return pd.Series(vif, index=descriptors.columns)


def step_vif_filter(descriptors):
    iteration = 1
    while descriptors.shape[1] > 1:
        vif_series = fast_vif(descriptors)
        max_vif = vif_series.max()
        if max_vif < VIF_THRESHOLD:
            break
        drop_feature = vif_series.idxmax()
        descriptors = descriptors.drop(columns=[drop_feature])
        print(f"[Step iii - VIF] Iteration {iteration}: dropped '{drop_feature}' "
              f"(VIF={max_vif:.2f}), remaining: {descriptors.shape[1]}")
        iteration += 1
    print(f"[Step iii - VIF] Final count (all VIF < {VIF_THRESHOLD}): {descriptors.shape[1]}")
    return descriptors


def main():
    metadata, target, descriptors_raw = load_and_split(INPUT_FILE)
    print(f"Loaded {len(metadata)} compounds. Target column: '{TARGET_COL}'\n")

    descriptors = basic_cleaning(descriptors_raw)

    # ---- (i) Variance filter ----
    descriptors = step_variance_filter(descriptors)
    df_variance = pd.concat([metadata.reset_index(drop=True),
                              descriptors.reset_index(drop=True)], axis=1)
    df_variance.to_csv(VARIANCE_OUT, index=False)
    print(f"[Step i] Saved: {VARIANCE_OUT}  shape={df_variance.shape}\n")

    # ---- (ii) Spearman filter ----
    descriptors = step_spearman_filter(descriptors, target)
    df_spearman = pd.concat([metadata.reset_index(drop=True),
                              descriptors.reset_index(drop=True)], axis=1)
    df_spearman.to_csv(SPEARMAN_OUT, index=False)
    print(f"[Step ii] Saved: {SPEARMAN_OUT}  shape={df_spearman.shape}\n")

    # ---- (iii) VIF filter ----
    descriptors = step_vif_filter(descriptors)
    df_vif = pd.concat([metadata.reset_index(drop=True),
                         descriptors.reset_index(drop=True)], axis=1)
    df_vif.to_csv(VIF_OUT, index=False)
    print(f"[Step iii] Saved: {VIF_OUT}  shape={df_vif.shape} "
          f"({descriptors.shape[1]} descriptors)\n")

    print("\n=========== SUMMARY ===========")
    print(f"Raw descriptors:            {descriptors_raw.shape[1]}")
    print(f"After (i) Variance filter:   -> see log above")
    print(f"After (ii) Spearman filter:  -> see log above")
    print(f"After (iii) VIF filter:      {descriptors.shape[1]}  <- saved as {VIF_OUT}")
    print(f"\nNext step: run boruta_rf_selection.py on {VIF_OUT} "
          f"(or your existing simple RF-importance script) to get the final descriptor set.")

    return df_variance, df_spearman, df_vif


if __name__ == "__main__":
    main()

#===========================================================================
