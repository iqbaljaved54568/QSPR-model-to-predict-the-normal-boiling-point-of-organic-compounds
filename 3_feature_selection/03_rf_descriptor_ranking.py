# Random forest descriptors ranking
#====================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FILE = "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/stage2_reduced.csv"
OUTPUT_RANKING_CSV = "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/rf_descriptor_ranking.csv"
OUTPUT_FIG_PATH = "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/rf_descriptor_ranking.png"

METADATA_ALIASES = [
    ["Name"], ["CAS No.", "CAS", "CAS_No", "CAS Number"],
    ["smiles", "SMILES", "canonical_smiles"],
    ["Formula", "Calculated_Formula"],
    ["molecular weight", "Molecular Weight", "MW"],
    ["boiling point (K)", "Boiling Point (K)", "BP", "boiling_point_K"],
]
TARGET_COL = "boiling point (K)"

N_ESTIMATORS = 1000
RANDOM_STATE = 42
TOP_N_TO_PLOT = 30
RUN_PERMUTATION_IMPORTANCE = True   # slower, but more reliable - see notes above
PERMUTATION_N_REPEATS = 10


def resolve_metadata_columns(df):
    rename_map, not_found = {}, []
    for aliases in METADATA_ALIASES:
        found = next((a for a in aliases if a in df.columns), None)
        if found is None:
            not_found.append(aliases)
        else:
            rename_map[found] = aliases[0]
    if not_found:
        raise ValueError(f"Could not find metadata fields: {not_found}. "
                          f"Available: {list(df.columns)[:15]}")
    return rename_map


def main():
    df = pd.read_csv(INPUT_FILE)
    df = df.rename(columns=resolve_metadata_columns(df))
    metadata_cols = [a[0] for a in METADATA_ALIASES]

    target = df[TARGET_COL].values
    descriptors = df.drop(columns=metadata_cols).apply(pd.to_numeric, errors="coerce").fillna(0)
    print(f"Loaded {len(df)} compounds, {descriptors.shape[1]} descriptors.")

    # ------------------------------------------------------------
    # Impurity-based importance (fit on all data - matches your
    # original Stage-3 approach)
    # ------------------------------------------------------------
    rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(descriptors, target)
    impurity_importance = pd.Series(rf.feature_importances_, index=descriptors.columns)

    ranking = pd.DataFrame({
        "descriptor": impurity_importance.index,
        "impurity_importance": impurity_importance.values,
    }).sort_values("impurity_importance", ascending=False).reset_index(drop=True)
    ranking["impurity_rank"] = ranking.index + 1
    ranking["cumulative_impurity_importance"] = ranking["impurity_importance"].cumsum()
    ranking["cumulative_pct"] = (ranking["cumulative_impurity_importance"]
                                  / ranking["impurity_importance"].sum() * 100).round(2)

    print("\nTop 15 by impurity importance:")
    print(ranking[["descriptor", "impurity_importance", "cumulative_pct"]].head(15)
          .to_string(index=False))

    # ------------------------------------------------------------
    # Permutation importance (held-out split, more reliable)
    # ------------------------------------------------------------
    if RUN_PERMUTATION_IMPORTANCE:
        print("\nComputing permutation importance (held-out split)...")
        X_train, X_test, y_train, y_test = train_test_split(
            descriptors, target, test_size=0.2, random_state=RANDOM_STATE
        )
        rf_perm = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
        rf_perm.fit(X_train, y_train)

        perm_result = permutation_importance(
            rf_perm, X_test, y_test,
            n_repeats=PERMUTATION_N_REPEATS, random_state=RANDOM_STATE, n_jobs=-1
        )
        perm_importance = pd.Series(perm_result.importances_mean, index=descriptors.columns)
        perm_std = pd.Series(perm_result.importances_std, index=descriptors.columns)

        ranking["permutation_importance"] = ranking["descriptor"].map(perm_importance)
        ranking["permutation_importance_std"] = ranking["descriptor"].map(perm_std)
        ranking = ranking.sort_values("impurity_importance", ascending=False).reset_index(drop=True)

        # Check agreement between the two ranking methods
        top20_impurity = set(ranking.sort_values("impurity_importance", ascending=False)
                              ["descriptor"].head(20))
        top20_perm = set(ranking.sort_values("permutation_importance", ascending=False)
                          ["descriptor"].head(20))
        overlap = len(top20_impurity & top20_perm)
        print(f"\nTop-20 agreement between impurity and permutation importance: "
              f"{overlap}/20 descriptors in common")

        print("\nTop 15 by permutation importance:")
        print(ranking.sort_values("permutation_importance", ascending=False)
              [["descriptor", "permutation_importance", "permutation_importance_std"]]
              .head(15).to_string(index=False))

    ranking.to_csv(OUTPUT_RANKING_CSV, index=False)
    print(f"\nFull ranking saved: {OUTPUT_RANKING_CSV}")

    # ------------------------------------------------------------
    # Plot: top N by impurity importance, with permutation importance
    # overlaid if available
    # ------------------------------------------------------------
    top = ranking.head(TOP_N_TO_PLOT).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.32 * TOP_N_TO_PLOT + 1.5))
    ax.barh(top["descriptor"], top["impurity_importance"], color="#2C6E8A",
            label="Impurity importance", height=0.7)
    if RUN_PERMUTATION_IMPORTANCE:
        ax2 = ax.twiny()
        ax2.plot(top["permutation_importance"], top["descriptor"], "o", color="#C0392B",
                 markersize=4, label="Permutation importance")
        ax2.set_xlabel("Permutation importance", color="#C0392B")
        ax2.tick_params(axis="x", colors="#C0392B")
    ax.set_xlabel("Impurity importance (RF feature_importances_)")
    ax.set_title(f"Top {TOP_N_TO_PLOT} descriptors by Random Forest importance",
                 loc="left", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_PATH, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {OUTPUT_FIG_PATH}")

    return ranking


if __name__ == "__main__":
    main()

# =============================================================================
