"""
Checks the layout of the train/test files

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

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
    "train_97desc": "./BP_FINAL_MODEL/descriptors/train_test_split/train_97desc.csv",
    "test_97desc":  "./BP_FINAL_MODEL/descriptors/train_test_split/test_97desc.csv",
    "train_35desc": "./BP_FINAL_MODEL/descriptors/train_test_split/train_38desc.csv",
    "test_35desc":  "./BP_FINAL_MODEL/descriptors/train_test_split/test_38desc.csv",
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

    out_path = "./BP_FINAL_MODEL/descriptors/train_test_split/data_structure_summary.csv"
    summary[display_cols].to_csv(out_path, index=False)
    print(f"\nSaved summary table: {out_path}")

    return summary


if __name__ == "__main__":
    main()
