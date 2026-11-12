"""
Comparison of the 94- and 35-descriptor sets (Figure 5)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

"""
compare_descriptor_sets.py
================================================================================
Cross Descriptor-Set Comparison: 94-descriptor (VIF-filtered) vs 35-descriptor
(RF-importance-selected)

WHY THIS EXISTS
----------------
The main pipeline (BP_FINAL_MODEL) runs once per descriptor set and never
compares the two runs against each other, and it also assumes LightGBM is
the "best model" without checking. This script closes both gaps: it loads
the checkpoints saved by both runs, determines the genuinely best
(model, descriptor set, transform) combination across BOTH sets, and builds
the two requested comparison tables plus comparison figures — everything
needed to support (or refute) the claim that the 35-descriptor RF-selected
set performs comparably to (or better than) the 94-descriptor VIF-filtered
set, i.e. that VIF alone did not remove all the redundant/collinear signal.

PREREQUISITES
--------------
Run the main pipeline TWICE:
    ACTIVE_SET = "94desc"   -> saves .../figures/94desc/checkpoint_94desc.pkl
    ACTIVE_SET = "35desc"   -> saves .../figures/35desc/checkpoint_35desc.pkl

Each checkpoint must contain `results_raw` and `results_log` (all 5 models,
metrics + predictions, model objects stripped). If your checkpoint only has
`best_res` (the original pipeline's default), add this to Section 16b of the
main script, just before `joblib.dump(checkpoint, ...)`, and re-run both
descriptor sets once:

    def _strip_for_pickle(res):
        drop = {"model", "train_hist", "val_hist"}
        return {k: v for k, v in res.items() if k not in drop}
    checkpoint["results_raw"] = {n: _strip_for_pickle(r) for n, r in results_raw.items()}
    checkpoint["results_log"] = {n: _strip_for_pickle(r) for n, r in results_log.items()}
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =============================================================================
# 0. CONFIG — edit these paths to match your folder layout
# =============================================================================

OUT_DIR_ROOT = "./BP_FINAL_MODEL/figures/"
CHECKPOINTS = {
    "94desc": f"{OUT_DIR_ROOT}94desc/checkpoint_94desc.pkl",
    "35desc": f"{OUT_DIR_ROOT}35desc/checkpoint_35desc.pkl",
}
COMPARE_OUT_DIR = f"{OUT_DIR_ROOT}comparison_94_vs_35/"
os.makedirs(COMPARE_OUT_DIR, exist_ok=True)

MODEL_ORDER  = ["Linear", "RandomForest", "XGBoost", "LightGBM", "ANN"]
MODEL_COLORS = {
    "Linear": "#4A90D9", "RandomForest": "#27AE60",
    "XGBoost": "#E67E22", "LightGBM": "#C0392B", "ANN": "#8E44AD",
}
SET_HATCH = {"94desc": "", "35desc": "///"}
SET_ALPHA = {"94desc": 0.55, "35desc": 0.95}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.facecolor": "#F7F9FB", "figure.facecolor": "white",
    "grid.color": "#DDE3EA", "grid.linestyle": "--", "grid.linewidth": 0.55,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save_fig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  ✅  Saved → {path}")


# =============================================================================
# 1. LOAD BOTH CHECKPOINTS
# =============================================================================

ckpts = {}
for set_name, path in CHECKPOINTS.items():
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found for '{set_name}': {path}\n"
            f"Run the main pipeline with ACTIVE_SET='{set_name}' first."
        )
    ckpts[set_name] = joblib.load(path)
    if "results_raw" not in ckpts[set_name] or "results_log" not in ckpts[set_name]:
        raise KeyError(
            f"Checkpoint '{set_name}' has no results_raw/results_log. "
            f"See the module docstring for the two-line patch needed in Section 16b "
            f"of the main pipeline, then re-run that descriptor set."
        )
    print(f"  ✅  Loaded checkpoint: {set_name}  (p={ckpts[set_name]['p']} descriptors, "
          f"n_train={ckpts[set_name]['n_train']:,})")

# =============================================================================
# 2. BUILD THE TWO COMPARISON TABLES (raw numeric — not pre-formatted strings)
# =============================================================================


def build_cross_set_table(transform_key, transform_label):
    rows = []
    for set_name, ck in ckpts.items():
        res_dict = ck[transform_key]
        for model_name in MODEL_ORDER:
            r = res_dict[model_name]
            rows.append({
                "Descriptor Set": set_name,
                "Model": model_name,
                "CV R2 mean": r["cv_r2_mean"], "CV R2 std": r["cv_r2_std"],
                "CV MAE (K)": r["cv_mae"], "CV RMSE (K)": r["cv_rmse"],
                "Test R2": r["test_r2"], "Test MAE (K)": r["test_mae"],
                "Test RMSE (K)": r["test_rmse"], "Test MAPE (%)": r["test_mape"],
                "Test Bias (K)": r["test_bias"],
            })
    df = pd.DataFrame(rows)

    print(f"\n{'='*90}")
    print(f"  TABLE — {transform_label} scale, 94desc vs 35desc")
    print(f"{'='*90}")
    print(df.set_index(["Model", "Descriptor Set"]).round(4).to_string())

    csv_path = f"{COMPARE_OUT_DIR}table_{transform_key}_94_vs_35.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅  Saved → {csv_path}")

    print("\n  LaTeX:\n")
    print(df.set_index(["Model", "Descriptor Set"]).round(4).to_latex(
        caption=f"Model performance comparison, {transform_label.lower()} target, "
                f"94-descriptor (VIF-filtered) vs 35-descriptor (RF-selected) sets.",
        label=f"tab:{transform_key}_94_vs_35"
    ))
    return df


table_original = build_cross_set_table("results_raw", "Original")
table_log = build_cross_set_table("results_log", "Log")

# =============================================================================
# 3. GENUINE OVERALL WINNER — across ALL 5 models x 2 transforms x 2 sets
#    (20 combinations total; NOT assumed to be LightGBM or either descriptor set)
# =============================================================================

all_combo = pd.concat([
    table_original.assign(Transform="Original"),
    table_log.assign(Transform="Log"),
], ignore_index=True)

winner_row = all_combo.loc[all_combo["Test R2"].idxmax()]
print(f"\n{'='*90}")
print("  OVERALL WINNER (by Test R², across both descriptor sets, all models, both transforms)")
print(f"{'='*90}")
print(f"  Descriptor set : {winner_row['Descriptor Set']}")
print(f"  Model          : {winner_row['Model']}")
print(f"  Transform      : {winner_row['Transform']}")
print(f"  Test R²        : {winner_row['Test R2']:.4f}")
print(f"  Test MAE (K)   : {winner_row['Test MAE (K)']:.3f}")
print(f"  Test RMSE (K)  : {winner_row['Test RMSE (K)']:.3f}")

# Same-model, cross-set delta — the number that most directly speaks to the
# "does VIF leave collinearity behind" question: if 35desc matches or beats
# 94desc despite having ~1/3 the features, the extra 59 VIF-surviving
# descriptors in the 94-set were carrying redundant (collinear), not new,
# information.
delta_rows = []
for transform_key, label in [("results_raw", "Original"), ("results_log", "Log")]:
    for model_name in MODEL_ORDER:
        r94 = ckpts["94desc"][transform_key][model_name]
        r35 = ckpts["35desc"][transform_key][model_name]
        delta_rows.append({
            "Transform": label, "Model": model_name,
            "Test R2 (94desc)": r94["test_r2"], "Test R2 (35desc)": r35["test_r2"],
            "DeltaR2 (35 minus 94)": r35["test_r2"] - r94["test_r2"],
            "Test MAE (94desc)": r94["test_mae"], "Test MAE (35desc)": r35["test_mae"],
            "DeltaMAE (35 minus 94), K": r35["test_mae"] - r94["test_mae"],
        })
delta_df = pd.DataFrame(delta_rows)
print(f"\n{'='*90}")
print("  SAME-MODEL DELTA: 35desc vs 94desc")
print("  (positive DeltaR2 / negative DeltaMAE => 35desc as good or better)")
print(f"{'='*90}")
print(delta_df.round(4).to_string(index=False))
delta_df.to_csv(f"{COMPARE_OUT_DIR}table_same_model_delta_35_vs_94.csv", index=False)

n_r2_holds = (delta_df["DeltaR2 (35 minus 94)"] >= -0.01).sum()
print(f"\n  -> 35desc matches 94desc within 0.01 R2 (or exceeds it) in "
      f"{n_r2_holds}/{len(delta_df)} model x transform comparisons.")
print("     A high count here is the direct evidence that the descriptors dropped")
print("     between 94->35 were largely redundant given what VIF already removed —")
print("     i.e. VIF filtering alone did not eliminate all multicollinear/redundant")
print("     descriptors.")

# =============================================================================
# 4. FIGURE 1 — Grouped bars: 94 vs 35, all 5 models, both transforms
# =============================================================================


def plot_cross_set_bars(save_path=None):
    metrics = ["Test R2", "Test MAE (K)", "Test RMSE (K)", "Test MAPE (%)"]
    titles = ["Test R²", "Test MAE (K)", "Test RMSE (K)", "Test MAPE (%)"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Descriptor Set Comparison — 94 (VIF-filtered) vs 35 (RF-selected)",
                 fontsize=14, fontweight="bold", y=1.01)

    for row, (transform_df, t_label) in enumerate(
        [(table_original, "Original"), (table_log, "Log")]
    ):
        x = np.arange(len(MODEL_ORDER))
        w = 0.35
        for col, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[row, col]
            v94 = [transform_df[(transform_df["Model"] == m) &
                                 (transform_df["Descriptor Set"] == "94desc")][metric].values[0]
                   for m in MODEL_ORDER]
            v35 = [transform_df[(transform_df["Model"] == m) &
                                 (transform_df["Descriptor Set"] == "35desc")][metric].values[0]
                   for m in MODEL_ORDER]
            ax.bar(x - w / 2, v94, w, color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                   alpha=SET_ALPHA["94desc"], edgecolor="white")
            ax.bar(x + w / 2, v35, w, color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                   alpha=SET_ALPHA["35desc"], edgecolor="white", hatch=SET_HATCH["35desc"])
            ax.set_xticks(x)
            ax.set_xticklabels(MODEL_ORDER, rotation=25, ha="right", fontsize=8)
            ax.set_title(f"{title} — {t_label}", fontsize=9.5, fontstyle="italic")
            ax.yaxis.grid(True)
            ax.set_axisbelow(True)

    legend_elem = [Patch(facecolor="#888", alpha=0.55, label="94desc (VIF-filtered)"),
                   Patch(facecolor="#888", alpha=0.95, hatch="///", label="35desc (RF-selected)")]
    fig.legend(handles=legend_elem, loc="upper right", bbox_to_anchor=(0.99, 0.985), fontsize=10)
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.show()
    plt.close(fig)


plot_cross_set_bars(save_path=f"{COMPARE_OUT_DIR}fig_cross_set_comparison.png")

# =============================================================================
# 5. FIGURE 2 — Each set's own best model x transform: pred vs exp, side by side
# =============================================================================


def best_of_set(set_name):
    # ignore_index=True is essential here: table_original and table_log each
    # carry their own independent RangeIndex, so filtering both by the same
    # set_name and concatenating without resetting the index produces
    # DUPLICATE index labels. df.loc[idxmax()] then returns a 2-row
    # DataFrame instead of a single Series, and row["Transform"] becomes a
    # Series instead of a string -- which is what caused the ValueError.
    df = pd.concat([
        table_original[table_original["Descriptor Set"] == set_name].assign(Transform="Original"),
        table_log[table_log["Descriptor Set"] == set_name].assign(Transform="Log"),
    ], ignore_index=True)
    row = df.loc[df["Test R2"].idxmax()]
    return row["Model"], row["Transform"]


fig2, axes2 = plt.subplots(1, 2, figsize=(13, 6.2))
fig2.suptitle("Best Model per Descriptor Set — Predicted vs Experimental (Test Set)",
              fontsize=13, fontweight="bold", y=1.02)

for ax, set_name in zip(axes2, ["94desc", "35desc"]):
    model_name, transform = best_of_set(set_name)
    res_key = "results_log" if transform == "Log" else "results_raw"
    res = ckpts[set_name][res_key][model_name]
    y_test = ckpts[set_name]["y_test"]
    y_pred = res["y_pred_test"]

    ax.scatter(y_test, y_pred, s=14, alpha=0.55, color=MODEL_COLORS[model_name], edgecolors="none")
    lims = [min(y_test.min(), y_pred.min()) - 5, max(y_test.max(), y_pred.max()) + 5]
    ax.plot(lims, lims, "--", color="#2C3E50", linewidth=1.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.text(0.04, 0.96,
            f"{model_name}{' (log)' if transform == 'Log' else ''}\n"
            f"R² = {res['test_r2']:.4f}\nMAE = {res['test_mae']:.2f} K\nRMSE = {res['test_rmse']:.2f} K",
            transform=ax.transAxes, fontsize=9.5, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#C5CDD5"))
    ax.set_xlabel("Experimental BP (K)", fontsize=10)
    ax.set_ylabel("Predicted BP (K)", fontsize=10)
    ax.set_title(f"{set_name}  (p={ckpts[set_name]['p']} descriptors)", fontsize=11, fontweight="bold")
    ax.grid(True)
    ax.set_axisbelow(True)

plt.tight_layout()
save_fig(fig2, f"{COMPARE_OUT_DIR}fig_best_model_per_set_pred_vs_exp.png")
plt.show()
plt.close(fig2)

# =============================================================================
# 6. FIGURE 3 — Log vs Original, for the OVERALL winning descriptor set only
# =============================================================================

winning_set = winner_row["Descriptor Set"]
fig3, ax3 = plt.subplots(figsize=(9, 6))
x = np.arange(len(MODEL_ORDER))
w = 0.35
r2_orig = [ckpts[winning_set]["results_raw"][m]["test_r2"] for m in MODEL_ORDER]
r2_log = [ckpts[winning_set]["results_log"][m]["test_r2"] for m in MODEL_ORDER]
ax3.bar(x - w / 2, r2_orig, w, color=[MODEL_COLORS[m] for m in MODEL_ORDER], alpha=0.55,
        edgecolor="white", label="Original")
ax3.bar(x + w / 2, r2_log, w, color=[MODEL_COLORS[m] for m in MODEL_ORDER], alpha=0.95,
        edgecolor="white", hatch="///", label="Log")
ax3.set_xticks(x)
ax3.set_xticklabels(MODEL_ORDER, rotation=20, ha="right")
ax3.set_ylabel("Test R²", fontsize=10)
ax3.set_title(f"Log vs Original Target — Winning Descriptor Set ({winning_set})",
              fontsize=12, fontweight="bold")
ax3.legend(fontsize=9)
ax3.yaxis.grid(True)
ax3.set_axisbelow(True)
plt.tight_layout()
save_fig(fig3, f"{COMPARE_OUT_DIR}fig_log_vs_original_winning_set.png")
plt.show()
plt.close(fig3)

print(f"\n{'='*90}")
print("  CROSS-DESCRIPTOR-SET COMPARISON COMPLETE")
print(f"{'='*90}")
print(f"""
  Outputs saved to: {COMPARE_OUT_DIR}
  |-- table_results_raw_94_vs_35.csv        <- Table 1 (Original scale, 94 vs 35, numeric)
  |-- table_results_log_94_vs_35.csv        <- Table 2 (Log scale, 94 vs 35, numeric)
  |-- table_same_model_delta_35_vs_94.csv   <- deltaR2 / deltaMAE per model, 35 minus 94
  |-- fig_cross_set_comparison.png          <- 2x4 grouped bars, all models, both transforms
  |-- fig_best_model_per_set_pred_vs_exp.png<- each set's own best model, pred vs exp
  `-- fig_log_vs_original_winning_set.png   <- log vs original, within the overall winner

  Overall winner: {winner_row['Model']} ({winner_row['Transform']}) on {winner_row['Descriptor Set']}
    Test R2 = {winner_row['Test R2']:.4f}
""")
