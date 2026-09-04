# Isomer Analysis — Final LightGBM Boiling Point Model (10 Optimal Descriptors)
# =============================================================================

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from scipy.stats import pearsonr, spearmanr, binomtest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

TRAIN_FILE = "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/train_test_split/train_optimal.csv"
TEST_FILE  = "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/train_test_split/test_optimal.csv"

TARGET_COL_INDEX       = 5
DESCRIPTOR_START_INDEX = 6

ACTIVE_SET = "10desc"
OUT_DIR    = f"/content/drive/MyDrive/BP_FINAL_MODEL/figures/{ACTIVE_SET}/"
os.makedirs(OUT_DIR, exist_ok=True)

SMILES_COL_FALLBACK_INDEX = 1   # only used if no recognizable SMILES column name is found

RANDOM_STATE = 42

# Minimum group size to be considered an isomer group worth analysing
MIN_ISOMER_GROUP_SIZE = 2
# Minimum experimental BP spread within a group to be "interesting" (filters
# out near-identical duplicates that add noise without adding information)
MIN_BP_SPREAD_K = 3.0

N_BOOTSTRAP = 2000   # bootstrap resamples for confidence intervals

# Final tuned LightGBM hyperparameters (10-descriptor optimal set) -- same
# values used in the Y-randomisation and full pipeline scripts, so this
# analysis reflects the exact same model reported everywhere else.
LGBM_PARAMS = {
    "n_estimators":      1784,
    "learning_rate":     0.06686808497475842,
    "num_leaves":        18,
    "max_depth":         6,
    "min_child_samples": 32,
    "subsample":         0.9921028774475024,
    "colsample_bytree":  0.7660499104221706,
    "reg_alpha":         0.06008952920081579,
    "reg_lambda":        0.001212337250189692,
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
    "verbosity":         -1,
}

# =============================================================================
# 2. STYLE
# =============================================================================

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.edgecolor":    "#B0BEC5",
    "axes.linewidth":    0.8,
    "axes.facecolor":    "#F7F9FB",
    "figure.facecolor":  "white",
    "grid.color":        "#DDE3EA",
    "grid.linewidth":    0.55,
    "grid.linestyle":    "--",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8.5,
    "legend.framealpha": 0.92,
    "legend.edgecolor":  "#C5CDD5",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

def save_and_show(fig, fname):
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  ✅  Saved → {path}")
    try:
        from IPython.display import Image as IPImage, display
        display(IPImage(filename=path))
    except Exception:
        pass
    plt.close(fig)

# =============================================================================
# 3. LOAD DATA + FIT THE MODEL DIRECTLY — no checkpoint dependency
# =============================================================================

print("=" * 65)
print(f"  Isomer Analysis — Final LightGBM Model | {ACTIVE_SET}")
print("=" * 65)

print(f"\n  Train file: {TRAIN_FILE}")
print(f"  Test file : {TEST_FILE}")

train_df = pd.read_csv(TRAIN_FILE)
test_df  = pd.read_csv(TEST_FILE).reset_index(drop=True)

X_train = train_df.iloc[:, DESCRIPTOR_START_INDEX:].values
y_train = train_df.iloc[:, TARGET_COL_INDEX].values
X_test  = test_df.iloc[:, DESCRIPTOR_START_INDEX:].values
y_test  = test_df.iloc[:, TARGET_COL_INDEX].values

n_train, p = X_train.shape
print(f"  Train: {n_train:,} x {p}  |  Test: {X_test.shape[0]:,} x {p}"
      f"  |  BP: {y_train.min():.0f}-{y_train.max():.0f} K")

if p != 10:
    print(f"  ⚠  Warning: expected 10 descriptors for the optimal set, but found {p} "
          f"columns from DESCRIPTOR_START_INDEX={DESCRIPTOR_START_INDEX}. Double-check "
          f"TARGET_COL_INDEX / DESCRIPTOR_START_INDEX against train_optimal.csv.")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n  Fitting LightGBM with locked {ACTIVE_SET} hyperparameters ...")
model = LGBMRegressor(**LGBM_PARAMS)
model.fit(X_train_sc, y_train)
y_pred = model.predict(X_test_sc)
tag_str = ""   # raw target, no log transform -- matches the Y-randomisation script

overall_r2   = r2_score(y_test, y_pred)
overall_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
overall_mae  = mean_absolute_error(y_test, y_pred)

print(f"  ✅  Model fit — LightGBM{tag_str}  n_test={len(y_test):,}")
print(f"  Overall model performance:")
print(f"    R²={overall_r2:.4f}  RMSE={overall_rmse:.2f} K  MAE={overall_mae:.2f} K")

# =============================================================================
# 4. COMPUTE MOLECULAR FORMULA FOR EVERY TEST COMPOUND
#    Uses RDKit to get the molecular formula from the SMILES string.
#    Compounds with the same formula but different SMILES are isomers.
# =============================================================================

print("\n  Computing molecular formulas from SMILES ...")

smiles_col = None
for candidate in ["smiles", "SMILES", "Smiles", "canonical_smiles", "smi"]:
    if candidate in test_df.columns:
        smiles_col = candidate
        break
if smiles_col is None:
    smiles_col = test_df.columns[SMILES_COL_FALLBACK_INDEX]
    print(f"  ⚠  No recognizable SMILES column name found — falling back to "
          f"column index {SMILES_COL_FALLBACK_INDEX} ('{smiles_col}'). "
          f"Verify this is actually the SMILES column.")

print(f"  Using SMILES column: '{smiles_col}'")

test_df["y_actual"]    = y_test
test_df["y_predicted"] = y_pred
test_df["abs_error"]   = np.abs(y_test - y_pred)
test_df["residual"]    = y_test - y_pred

mol_formulas, mol_weights, valid_mask = [], [], []
for smi in test_df[smiles_col]:
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            mol_formulas.append(rdMolDescriptors.CalcMolFormula(mol))
            mol_weights.append(round(Descriptors.MolWt(mol), 3))
            valid_mask.append(True)
        else:
            mol_formulas.append("INVALID"); mol_weights.append(np.nan); valid_mask.append(False)
    except Exception:
        mol_formulas.append("INVALID"); mol_weights.append(np.nan); valid_mask.append(False)

test_df["mol_formula"]  = mol_formulas
test_df["mol_weight"]   = mol_weights
test_df["valid_smiles"] = valid_mask

n_valid = sum(valid_mask)
print(f"  Valid SMILES: {n_valid:,} / {len(test_df):,}")

# =============================================================================
# 5. IDENTIFY ISOMER GROUPS
# =============================================================================

print("\n  Identifying isomer groups ...")
valid_df = test_df[test_df["valid_smiles"]].copy()
formula_groups = valid_df.groupby("mol_formula")

isomer_groups = {}
for formula, group in formula_groups:
    if formula == "INVALID":
        continue
    if group[smiles_col].nunique() < MIN_ISOMER_GROUP_SIZE:
        continue
    bp_spread = group["y_actual"].max() - group["y_actual"].min()
    if bp_spread < MIN_BP_SPREAD_K:
        continue
    isomer_groups[formula] = group.copy()

n_groups      = len(isomer_groups)
n_isomer_cpds = sum(len(g) for g in isomer_groups.values())
group_sizes   = [len(g) for g in isomer_groups.values()]

print(f"\n  Isomer groups found       : {n_groups}")
print(f"  Total compounds in groups : {n_isomer_cpds}")
print(f"  Group size distribution   :")
for sz in sorted(set(group_sizes)):
    print(f"    {sz:2d} isomers : {group_sizes.count(sz):3d} groups")

if n_groups == 0:
    raise SystemExit("No isomer groups found — check MIN_BP_SPREAD_K / SMILES column / CSV contents.")

# =============================================================================
# 6. ISOMER DISCRIMINATION STATISTICS (per-group descriptive stats)
# =============================================================================

print("\n  Computing isomer discrimination statistics ...")

group_stats = []
all_exp_spreads, all_pred_spreads = [], []
correct_direction_count, total_pairs = 0, 0
group_pair_correct = []   # per-group (correct, total) — needed for bootstrap-by-group

for formula, group in isomer_groups.items():
    g = group.sort_values("y_actual").reset_index(drop=True)
    exp_bp, pred_bp = g["y_actual"].values, g["y_predicted"].values
    exp_spread, pred_spread = exp_bp.max() - exp_bp.min(), pred_bp.max() - pred_bp.min()

    if len(g) >= 3:
        rho, _ = spearmanr(exp_bp, pred_bp)
    else:
        rho = 1.0 if (exp_bp[1]-exp_bp[0])*(pred_bp[1]-pred_bp[0]) > 0 else -1.0

    n = len(g)
    g_correct, g_total = 0, 0
    for i in range(n):
        for j in range(i+1, n):
            g_total += 1
            if (exp_bp[j]-exp_bp[i]) * (pred_bp[j]-pred_bp[i]) > 0:
                g_correct += 1
    correct_direction_count += g_correct
    total_pairs += g_total
    group_pair_correct.append((g_correct, g_total))

    all_exp_spreads.append(exp_spread)
    all_pred_spreads.append(pred_spread)
    group_stats.append({
        "Formula": formula, "N_isomers": len(g), "MW": g["mol_weight"].mean(),
        "Exp_BP_min": exp_bp.min(), "Exp_BP_max": exp_bp.max(),
        "Exp_spread_K": exp_spread, "Pred_spread_K": pred_spread,
        "Spread_ratio": pred_spread/exp_spread if exp_spread > 0 else 0,
        "Spearman_rho": rho, "MAE_in_group": np.abs(exp_bp-pred_bp).mean(),
        "SMILES_list": g[smiles_col].tolist(),
        "Exp_BPs": exp_bp.tolist(), "Pred_BPs": pred_bp.tolist(),
    })

stats_df = pd.DataFrame(group_stats).sort_values("Exp_spread_K", ascending=False).reset_index(drop=True)
pairwise_accuracy = correct_direction_count / total_pairs * 100 if total_pairs > 0 else 0

# =============================================================================
# 6b. STATISTICAL SIGNIFICANCE — is this better than the exact 50% null?
#     Binomial test: since every isomer in a group shares an identical
#     molecular formula (hence identical MW), a structure-blind model has an
#     EXACT null expectation of 50% pairwise accuracy — not an assumption,
#     a theoretical fact. This is directly analogous to the Y-randomisation
#     test elsewhere in this project: both ask "could this have happened by
#     chance alone?", just with a different, more specific null here.
#
#     Bootstrap CI resamples whole GROUPS (not individual pairs), because
#     pairs within the same group are correlated (they share isomers), so
#     treating them as independent would understate the true uncertainty.
# =============================================================================

print(f"\n  {'='*60}")
print(f"  STATISTICAL SIGNIFICANCE — pairwise ranking accuracy vs 50% null")
print(f"  {'='*60}")

binom_result = binomtest(correct_direction_count, total_pairs, p=0.5, alternative="greater")
print(f"  Correct / total pairs        : {correct_direction_count:,} / {total_pairs:,}  "
      f"({pairwise_accuracy:.1f}%)")
print(f"  Binomial test vs 50% null    : p = {binom_result.pvalue:.2e}")
print(f"  95% CI on true accuracy      : "
      f"[{binom_result.proportion_ci(confidence_level=0.95).low*100:.1f}%, "
      f"{binom_result.proportion_ci(confidence_level=0.95).high*100:.1f}%]")

rng = np.random.RandomState(RANDOM_STATE)
n_g = len(group_pair_correct)
boot_accuracies, boot_rhos = [], []
rho_values_arr = stats_df["Spearman_rho"].values
for _ in range(N_BOOTSTRAP):
    idx = rng.randint(0, n_g, size=n_g)
    c = sum(group_pair_correct[i][0] for i in idx)
    t = sum(group_pair_correct[i][1] for i in idx)
    boot_accuracies.append(c / t * 100 if t > 0 else np.nan)
    boot_rhos.append(np.nanmean(rho_values_arr[idx]))

boot_accuracies = np.array(boot_accuracies)
boot_rhos       = np.array(boot_rhos)
acc_ci  = np.nanpercentile(boot_accuracies, [2.5, 97.5])
rho_ci  = np.nanpercentile(boot_rhos, [2.5, 97.5])

print(f"\n  Group-level bootstrap ({N_BOOTSTRAP} resamples, resampling by GROUP):")
print(f"  Pairwise accuracy 95% CI    : [{acc_ci[0]:.1f}%, {acc_ci[1]:.1f}%]")
print(f"  Mean Spearman ρ 95% CI      : [{rho_ci[0]:.3f}, {rho_ci[1]:.3f}]")
if acc_ci[0] > 50:
    print(f"  ✅  Even the LOWER bound of the 95% CI ({acc_ci[0]:.1f}%) exceeds the")
    print(f"      50% chance baseline — discrimination is robust, not a fluke of")
    print(f"      which groups happened to be in the test set.")
else:
    print(f"  ⚠  Lower CI bound touches or crosses 50% — discrimination claim is")
    print(f"      weaker than the point estimate suggests; consider more isomer")
    print(f"      groups or reporting this caveat explicitly.")

print(f"\n  Mean Spearman rho (within group): {stats_df['Spearman_rho'].mean():.3f}")
print(f"  Mean experimental spread        : {np.mean(all_exp_spreads):.1f} K")
print(f"  Mean predicted spread           : {np.mean(all_pred_spreads):.1f} K")
print(f"  Spread ratio (pred/exp)         : {np.mean(stats_df['Spread_ratio']):.3f}  "
      f"(1.0 = perfect; < 1 = underestimation)")

print(f"\n  Top 10 isomer groups (largest experimental BP spread):\n")
print(f"  {'Formula':<12} {'N':>3} {'MW':>7} {'Exp spread':>11} {'Pred spread':>12} "
      f"{'Spread ratio':>13} {'Spearman ρ':>11}")
print(f"  {'─'*75}")
for _, row in stats_df.head(10).iterrows():
    print(f"  {row['Formula']:<12} {row['N_isomers']:>3} {row['MW']:>7.1f} "
          f"{row['Exp_spread_K']:>9.1f} K{row['Pred_spread_K']:>11.1f} K"
          f"{row['Spread_ratio']:>11.3f}{row['Spearman_rho']:>11.3f}")

# =============================================================================
# 7. FIGURE 1 — Experimental vs Predicted BP for Isomer Groups
# =============================================================================

fig1, axes = plt.subplots(1, 2, figsize=(14, 6))
fig1.suptitle(f"Isomer Discrimination — LightGBM{tag_str} | {ACTIVE_SET}\n"
              "Model predictions for constitutional isomers in the test set",
              fontsize=12, fontweight="bold", y=1.02)

ax = axes[0]
ax.scatter(y_test, y_pred, s=8, alpha=0.15, color="#B4B2A9",
           edgecolors="none", rasterized=True, label="All test compounds")

cmap = plt.cm.plasma
sizes = stats_df["N_isomers"].values
sz_min, sz_max = 2, max(sizes)
for _, row in stats_df.iterrows():
    for exp_bp, pred_bp in zip(row["Exp_BPs"], row["Pred_BPs"]):
        norm_sz = (row["N_isomers"] - sz_min) / max(sz_max - sz_min, 1)
        ax.scatter(exp_bp, pred_bp, s=25, alpha=0.75,
                   color=cmap(0.2 + 0.75*norm_sz), edgecolors="white",
                   linewidths=0.3, zorder=5, rasterized=True)

lims = [min(y_test.min(), y_pred.min())-5, max(y_test.max(), y_pred.max())+5]
ax.plot(lims, lims, "k--", lw=1.4, label="y = x", zorder=3)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=sz_min, vmax=sz_max))
sm.set_array([])
cb = plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
cb.set_label("Isomer group size", fontsize=8.5); cb.ax.tick_params(labelsize=8)

ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
ax.set_xlabel("Experimental BP (K)", fontsize=10)
ax.set_ylabel("Predicted BP (K)", fontsize=10)
ax.set_title("(A)  Predicted vs Experimental — isomer compounds highlighted",
             loc="left", fontsize=9.5, fontstyle="italic")
ax.text(0.05, 0.95,
        f"Isomer groups: {n_groups}\n"
        f"Compounds in groups: {n_isomer_cpds}\n"
        f"Pairwise accuracy: {pairwise_accuracy:.1f}%\n"
        f"(95% CI: {acc_ci[0]:.1f}–{acc_ci[1]:.1f}%; p={binom_result.pvalue:.1e} vs 50%)",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#C5CDD5"))
ax.legend(fontsize=8, loc="lower right")
ax.grid(True); ax.set_axisbelow(True)

ax2 = axes[1]
scatter = ax2.scatter(stats_df["Exp_spread_K"], stats_df["Pred_spread_K"],
                      c=stats_df["N_isomers"], cmap="plasma", vmin=sz_min, vmax=sz_max,
                      s=40, alpha=0.75, edgecolors="white", linewidths=0.5, zorder=5)
max_sp = max(stats_df["Exp_spread_K"].max(), stats_df["Pred_spread_K"].max()) * 1.1
ax2.plot([0, max_sp], [0, max_sp], "k--", lw=1.4, label="Ideal (1:1)", zorder=3)

r_val, p_val = pearsonr(stats_df["Exp_spread_K"], stats_df["Pred_spread_K"])
ax2.text(0.05, 0.95,
         f"Pearson r = {r_val:.3f}\nn groups = {n_groups}\n"
         f"Mean spread ratio = {stats_df['Spread_ratio'].mean():.3f}",
         transform=ax2.transAxes, fontsize=8.5, va="top",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#C5CDD5"))
ax2.set_xlabel("Experimental BP spread within group (K)", fontsize=10)
ax2.set_ylabel("Predicted BP spread within group (K)", fontsize=10)
ax2.set_title("(B)  Experimental vs predicted BP spread per isomer group",
              loc="left", fontsize=9.5, fontstyle="italic")
ax2.legend(fontsize=8.5); ax2.grid(True); ax2.set_axisbelow(True)
plt.colorbar(scatter, ax=ax2, shrink=0.75, pad=0.02).set_label("Isomer group size", fontsize=8.5)

plt.tight_layout()
save_and_show(fig1, f"fig_isomer_scatter_{ACTIVE_SET}.png")

# =============================================================================
# 8. FIGURE 2 — Top 12 Isomer Groups (individual group plots)
# =============================================================================

top_n = min(12, n_groups)
ncols = 4
nrows = (top_n + ncols - 1) // ncols

fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols*3.6, nrows*3.2))
fig2.suptitle(f"Boiling Point Predictions for Top {top_n} Isomer Groups ({ACTIVE_SET})\n"
              "Blue = experimental, orange = predicted, ordered by experimental BP spread",
              fontsize=11, fontweight="bold", y=1.01)
axes2_flat = axes2.flat if nrows > 1 else ([axes2] if top_n == 1 else axes2)

for idx, (ax_g, (_, row)) in enumerate(zip(axes2_flat, stats_df.head(top_n).iterrows())):
    exp_bps, pred_bps = np.array(row["Exp_BPs"]), np.array(row["Pred_BPs"])
    n_iso = row["N_isomers"]
    order = np.argsort(exp_bps)
    exp_ord, pred_ord = exp_bps[order], pred_bps[order]
    x, width = np.arange(n_iso), 0.38

    bars_exp  = ax_g.bar(x-width/2, exp_ord, width, color="#2C6E8A", alpha=0.85,
                        edgecolor="white", linewidth=0.5, label="Experimental")
    bars_pred = ax_g.bar(x+width/2, pred_ord, width, color="#E67E22", alpha=0.85,
                        edgecolor="white", linewidth=0.5, label="Predicted")
    for bar, val in zip(bars_exp, exp_ord):
        ax_g.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{val:.0f}",
                  ha="center", va="bottom", fontsize=6.5, color="#2C6E8A", fontweight="bold")
    for bar, val in zip(bars_pred, pred_ord):
        ax_g.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{val:.0f}",
                  ha="center", va="bottom", fontsize=6.5, color="#E67E22")

    ax_g.set_title(f"{row['Formula']}  (n={n_iso})", fontsize=9, fontweight="bold")
    ax_g.set_xticks(x)
    ax_g.set_xticklabels([f"Iso {i+1}" for i in range(n_iso)], fontsize=7.5, rotation=25, ha="right")
    ax_g.set_ylabel("BP (K)", fontsize=8)
    ax_g.yaxis.grid(True, alpha=0.5); ax_g.set_axisbelow(True)
    ax_g.text(0.97, 0.04, f"ρ = {row['Spearman_rho']:.2f}\nΔ = {row['Exp_spread_K']:.0f} K",
              transform=ax_g.transAxes, fontsize=7, ha="right", va="bottom",
              bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#C5CDD5", alpha=0.9))
    if idx == 0:
        ax_g.legend(fontsize=7, loc="upper left")

for ax_g in list(axes2_flat)[top_n:]:
    ax_g.set_visible(False)

plt.tight_layout()
save_and_show(fig2, f"fig_isomer_groups_{ACTIVE_SET}.png")

# =============================================================================
# 9. FIGURE 3 — Spearman ρ / spread-ratio distributions + bootstrap CIs
# =============================================================================

fig3, (ax_rho, ax_ratio) = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle(f"Isomer Discrimination Statistics — Distribution across Groups ({ACTIVE_SET})",
              fontsize=12, fontweight="bold", y=1.02)

rho_vals = stats_df["Spearman_rho"].dropna().values
ax_rho.hist(rho_vals, bins=20, color="#2C6E8A", alpha=0.75, edgecolor="white", linewidth=0.5)
ax_rho.axvline(rho_vals.mean(), color="#C0392B", linewidth=2, linestyle="--",
               label=f"Mean ρ = {rho_vals.mean():.3f}  (95% CI {rho_ci[0]:.3f}–{rho_ci[1]:.3f})")
ax_rho.axvline(0, color="#AAB7B8", linewidth=1, linestyle=":")
ax_rho.set_xlabel("Spearman rank correlation (ρ) within group", fontsize=10)
ax_rho.set_ylabel("Number of isomer groups", fontsize=10)
ax_rho.set_title("(A)  Rank correlation distribution\nρ > 0 = model correctly ranks isomers by BP",
                 loc="left", fontsize=9.5, fontstyle="italic")
ax_rho.legend(fontsize=8)
ax_rho.text(0.05, 0.95,
            f"Groups with ρ > 0: {(rho_vals>0).sum()}/{len(rho_vals)} ({(rho_vals>0).mean()*100:.1f}%)\n"
            f"Groups with ρ > 0.5: {(rho_vals>0.5).sum()}/{len(rho_vals)} ({(rho_vals>0.5).mean()*100:.1f}%)\n"
            f"Groups with ρ = 1.0: {(rho_vals==1.0).sum()}/{len(rho_vals)} ({(rho_vals==1.0).mean()*100:.1f}%)",
            transform=ax_rho.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#C5CDD5"))
ax_rho.grid(True); ax_rho.set_axisbelow(True)

ratio_vals = stats_df["Spread_ratio"].values
ax_ratio.hist(ratio_vals, bins=20, color="#0F6E56", alpha=0.75, edgecolor="white", linewidth=0.5)
ax_ratio.axvline(ratio_vals.mean(), color="#C0392B", linewidth=2, linestyle="--",
                 label=f"Mean ratio = {ratio_vals.mean():.3f}")
ax_ratio.axvline(1.0, color="#2C3E50", linewidth=1.5, linestyle=":", label="Ideal (ratio = 1.0)")
ax_ratio.set_xlabel("Predicted spread / Experimental spread", fontsize=10)
ax_ratio.set_ylabel("Number of isomer groups", fontsize=10)
ax_ratio.set_title("(B)  BP spread ratio per isomer group\n1.0 = model captures full experimental range",
                   loc="left", fontsize=9.5, fontstyle="italic")
ax_ratio.legend(fontsize=8.5); ax_ratio.grid(True); ax_ratio.set_axisbelow(True)

plt.tight_layout()
save_and_show(fig3, f"fig_isomer_statistics_{ACTIVE_SET}.png")

# =============================================================================
# 9b. FIGURE 4 — Bootstrap distribution of pairwise accuracy vs 50% null
#     Makes the significance test visually explicit for the paper.
# =============================================================================

fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.hist(boot_accuracies, bins=40, color="#2C6E8A", alpha=0.75,
         edgecolor="white", linewidth=0.4, density=True,
         label=f"Bootstrap distribution (n={N_BOOTSTRAP} group resamples)")
ax4.axvline(50, color="#333", linewidth=1.8, linestyle="--", label="50% null (chance)")
ax4.axvline(pairwise_accuracy, color="#C0392B", linewidth=2.2,
            label=f"Observed = {pairwise_accuracy:.1f}%")
ax4.axvspan(acc_ci[0], acc_ci[1], alpha=0.15, color="#2C6E8A", label="95% CI")
ax4.set_xlabel("Pairwise Ranking Accuracy (%)", fontsize=10)
ax4.set_ylabel("Bootstrap Density", fontsize=10)
ax4.set_title(f"Isomer Pairwise Accuracy vs 50% Chance Null ({ACTIVE_SET})\n"
              f"p = {binom_result.pvalue:.2e} (binomial test)",
              fontsize=11, fontweight="bold")
ax4.legend(fontsize=8.5); ax4.grid(True); ax4.set_axisbelow(True)
plt.tight_layout()
save_and_show(fig4, f"fig_isomer_significance_{ACTIVE_SET}.png")

# =============================================================================
# 10. LaTeX TABLE — Top 10 isomer groups
# =============================================================================

latex_rows = []
for _, row in stats_df.head(10).iterrows():
    latex_rows.append({
        "Formula": row["Formula"], "N": int(row["N_isomers"]),
        "MW (g/mol)": f"{row['MW']:.1f}",
        "Exp. spread (K)": f"{row['Exp_spread_K']:.1f}",
        "Pred. spread (K)": f"{row['Pred_spread_K']:.1f}",
        "Spread ratio": f"{row['Spread_ratio']:.3f}",
        "Spearman $\\rho$": f"{row['Spearman_rho']:.3f}",
        "MAE (K)": f"{row['MAE_in_group']:.2f}",
    })
latex_df = pd.DataFrame(latex_rows)
print("\n  LaTeX table:\n")
print(latex_df.to_latex(
    index=False,
    caption=(f"Isomer discrimination results for the top 10 constitutional isomer "
             f"groups (ranked by experimental boiling point spread), {ACTIVE_SET}. "
             f"N = number of isomers; MW = mean molecular weight; Exp./Pred. spread "
             f"= experimental/predicted BP range within the group; Spread ratio = "
             f"Pred./Exp. spread (1.0 = perfect); Spearman $\\rho$ = rank correlation "
             f"of predicted vs experimental BP within the group; MAE = mean absolute "
             f"prediction error."),
    label=f"tab:isomer_analysis_{ACTIVE_SET}", escape=False,
))

# =============================================================================
# 11. SAVE FULL RESULTS CSV
# =============================================================================

results_csv = f"{OUT_DIR}isomer_analysis_results_{ACTIVE_SET}.csv"
stats_df.drop(columns=["SMILES_list","Exp_BPs","Pred_BPs"], errors="ignore").to_csv(results_csv, index=False)
print(f"\n  ✅  Full results saved → {results_csv}")

# =============================================================================
# 12. FINAL SUMMARY + PAPER TEXT
# =============================================================================

rho_pos_pct  = (rho_vals > 0).mean() * 100
rho_high_pct = (rho_vals > 0.5).mean() * 100
rho_perf_pct = (rho_vals == 1.0).mean() * 100

print(f"""
{'='*65}
  ISOMER ANALYSIS COMPLETE — {ACTIVE_SET}
{'='*65}

  Isomer groups identified     : {n_groups}
  Total isomer compounds       : {n_isomer_cpds}
  Total pairwise comparisons   : {total_pairs:,}
  Pairwise ranking accuracy    : {pairwise_accuracy:.1f}%  (95% CI {acc_ci[0]:.1f}–{acc_ci[1]:.1f}%)
  Binomial p-value vs 50% null : {binom_result.pvalue:.2e}
  Mean Spearman ρ              : {rho_vals.mean():.3f}  (95% CI {rho_ci[0]:.3f}–{rho_ci[1]:.3f})
  Groups with ρ > 0            : {rho_pos_pct:.1f}%
  Groups with ρ > 0.5          : {rho_high_pct:.1f}%
  Groups with ρ = 1.0 (perfect): {rho_perf_pct:.1f}%
  Mean spread ratio (pred/exp) : {ratio_vals.mean():.3f}

  Suggested paper paragraph:
  ────────────────────────────────────────────────────────────
  "To further evaluate the structural sensitivity of the final
  LightGBM model, constitutional isomers were identified within
  the external test set by grouping compounds sharing an
  identical molecular formula. A total of {n_groups} isomer
  groups comprising {n_isomer_cpds} compounds were identified,
  exhibiting experimental boiling point spreads ranging from
  {stats_df['Exp_spread_K'].min():.1f} K to {stats_df['Exp_spread_K'].max():.1f} K
  within groups. Because isomers within a group share an
  identical molecular formula — and therefore an identical
  molecular weight — a model with no genuine structural
  sensitivity has an exact theoretical null expectation of 50%
  pairwise ranking accuracy. The model correctly discriminated
  {pairwise_accuracy:.1f}% of all pairwise isomer rankings (95%
  CI {acc_ci[0]:.1f}–{acc_ci[1]:.1f}%, bootstrap resampled by
  isomer group), significantly exceeding the 50% chance baseline
  (binomial test, p = {binom_result.pvalue:.1e}), with a mean
  Spearman rank correlation of ρ = {rho_vals.mean():.3f} (95% CI
  {rho_ci[0]:.3f}–{rho_ci[1]:.3f}) across groups. The mean
  predicted BP spread ({ratio_vals.mean():.3f} × experimental
  spread) indicates that while the model captures the direction
  of isomeric differences reliably, it slightly underestimates
  the absolute magnitude of the spread — consistent with the
  known tendency of ensemble methods to regress toward the mean.
  These results confirm that the model encodes structural
  topology, branching, and functional group position rather than
  merely molecular composition."
  ────────────────────────────────────────────────────────────

  Figures saved:
    fig_isomer_scatter_{ACTIVE_SET}.png       Scatter + spread comparison
    fig_isomer_groups_{ACTIVE_SET}.png        Top {top_n} groups bar charts
    fig_isomer_statistics_{ACTIVE_SET}.png    ρ and spread ratio distributions
    fig_isomer_significance_{ACTIVE_SET}.png  Bootstrap accuracy vs 50% null
{'='*65}
""")