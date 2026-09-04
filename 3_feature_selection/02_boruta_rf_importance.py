# Boruta-style RF Importance Selection (Phase-2,Des_red)
#===========================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FILE = "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/stage2_reduced.csv"
SELECTED_OUT_TEMPLATE = "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/selected_{n}_descriptors.csv"

METADATA_ALIASES = [
    ["Name"],
    ["CAS No.", "CAS", "CAS_No", "CAS Number"],
    ["smiles", "SMILES", "canonical_smiles"],
    ["Formula", "Calculated_Formula"],
    ["molecular weight", "Molecular Weight", "MW"],
    ["boiling point (K)", "Boiling Point (K)", "BP", "boiling_point_K"],
]
TARGET_COL = "boiling point (K)"

# --- Speed vs. stability ---
FAST_MODE = True   # True = quick exploratory run, False = final/stable run

if FAST_MODE:
    N_ITERATIONS   = 20
    RF_N_ESTIMATORS = 150
else:
    N_ITERATIONS   = 50
    RF_N_ESTIMATORS = 300

HIT_THRESHOLD  = 0.60   # fraction of rounds a descriptor must "win" to be confirmed
RANDOM_STATE   = 42
RUN_SHAP_CHECK = True   # secondary ranking only, doesn't affect selection


def resolve_metadata_columns(df):
    rename_map, not_found = {}, []
    for aliases in METADATA_ALIASES:
        canonical = aliases[0]
        found = next((a for a in aliases if a in df.columns), None)
        if found is None:
            not_found.append(aliases)
        else:
            rename_map[found] = canonical
    if not_found:
        raise ValueError(f"Could not find metadata fields: {not_found}. "
                          f"Available: {list(df.columns)[:15]}")
    return rename_map


def load_and_split(path):
    df = pd.read_csv(path)
    df = df.rename(columns=resolve_metadata_columns(df))
    metadata_cols = [a[0] for a in METADATA_ALIASES]
    metadata = df[metadata_cols].copy()
    target = df[TARGET_COL].copy()
    descriptors = df.drop(columns=metadata_cols).apply(pd.to_numeric, errors="coerce").fillna(0)
    return metadata, target, descriptors


def boruta_style_selection(descriptors, target,
                            n_iterations=N_ITERATIONS,
                            hit_threshold=HIT_THRESHOLD,
                            rf_n_estimators=RF_N_ESTIMATORS,
                            random_state=RANDOM_STATE):
    rng = np.random.RandomState(random_state)
    feature_names = descriptors.columns.tolist()
    win_counts = pd.Series(0, index=feature_names)
    importance_accum = pd.Series(0.0, index=feature_names)

    X_real = descriptors.values

    for it in range(n_iterations):
        shadow = X_real.copy()
        for j in range(shadow.shape[1]):
            rng.shuffle(shadow[:, j])

        X_combined = np.hstack([X_real, shadow])
        combined_names = feature_names + [f"shadow_{f}" for f in feature_names]

        rf = RandomForestRegressor(
            n_estimators=rf_n_estimators,
            random_state=rng.randint(0, 1_000_000),
            n_jobs=-1,
        )
        rf.fit(X_combined, target)
        importances = pd.Series(rf.feature_importances_, index=combined_names)

        real_importances = importances[feature_names]
        shadow_importances = importances[[f"shadow_{f}" for f in feature_names]]
        shadow_max = shadow_importances.max()

        wins = real_importances > shadow_max
        win_counts += wins.astype(int)
        importance_accum += real_importances

        print(f"[Boruta/RF] Iteration {it + 1}/{n_iterations} "
              f"(shadow max importance this round: {shadow_max:.5f})")

    win_rate = win_counts / n_iterations
    mean_importance = importance_accum / n_iterations

    confirmed = win_rate[win_rate >= hit_threshold].index.tolist()
    confirmed_sorted = mean_importance[confirmed].sort_values(ascending=False)

    print(f"\n[Boruta/RF] {len(confirmed)} descriptors confirmed "
          f"(win rate >= {hit_threshold:.0%} across {n_iterations} iterations)")
    for f in confirmed_sorted.index:
        print(f"    {f:45s} win_rate={win_rate[f]:.2f}  mean_importance={mean_importance[f]:.5f}")

    summary = pd.DataFrame({
        "win_rate": win_rate,
        "mean_rf_importance": mean_importance,
    }).sort_values("mean_rf_importance", ascending=False)

    return confirmed_sorted.index.tolist(), summary


def add_shap_ranking(descriptors_confirmed, target, random_state=RANDOM_STATE):
    """Optional secondary ranking via SHAP on the confirmed set only.
    Reporting/QA - does not affect which descriptors were selected."""
    if not SHAP_AVAILABLE:
        print("\n[SHAP] 'shap' package not installed - skipping. "
              "Install with: pip install shap --break-system-packages")
        return None

    rf = RandomForestRegressor(n_estimators=500, random_state=random_state, n_jobs=-1)
    rf.fit(descriptors_confirmed, target)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(descriptors_confirmed)
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=descriptors_confirmed.columns
    ).sort_values(ascending=False)

    print("\n[SHAP] Mean |SHAP value| ranking (confirmed descriptors only):")
    for f, v in mean_abs_shap.items():
        print(f"    {f:45s} {v:.5f}")
    return mean_abs_shap


def main():
    metadata, target, descriptors = load_and_split(INPUT_FILE)
    print(f"Loaded {len(metadata)} compounds, {descriptors.shape[1]} descriptors "
          f"from {INPUT_FILE}")
    print(f"Mode: {'FAST (exploratory)' if FAST_MODE else 'FULL (final/stable)'}  "
          f"-> N_ITERATIONS={N_ITERATIONS}, RF_N_ESTIMATORS={RF_N_ESTIMATORS}\n")

    confirmed_features, summary = boruta_style_selection(descriptors, target)
    descriptors_selected = descriptors[confirmed_features]
    n_selected = len(confirmed_features)

    if RUN_SHAP_CHECK:
        add_shap_ranking(descriptors_selected, target)

    df_selected = pd.concat([metadata.reset_index(drop=True),
                              descriptors_selected.reset_index(drop=True)], axis=1)
    selected_out = SELECTED_OUT_TEMPLATE.format(n=n_selected)
    df_selected.to_csv(selected_out, index=False)

    summary_out = SELECTED_OUT_TEMPLATE.format(n=f"{n_selected}_summary")
    summary.to_csv(summary_out)

    print(f"\nSaved: {selected_out}  shape={df_selected.shape} "
          f"({n_selected} descriptors, automatically determined)")
    print(f"Saved selection summary (win rates, importances): {summary_out}")

    return df_selected, summary


if __name__ == "__main__":
    main()

#===================================
