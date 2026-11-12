"""
SHAP interactions, Friedman H, additive model (Table S19, Figure S11)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# Quantifying descriptor interactions in the final 10-descriptor LightGBM model
#
#  1. Additive-vs-interaction test: refit LightGBM with identical hyperparameters but
#     interactions FORBIDDEN (each tree may split on one descriptor only -> additive
#     model). The loss in CV/test accuracy = contribution of interactions.
#  2. SHAP interaction values (training set): share of total attribution due to
#     pairwise interactions, and the 10x10 interaction-strength matrix.
#  3. Friedman's H-statistic for the pairs among the top three descriptors.
#  4. Figure: 2D partial dependence for the three top-descriptor pairs + centred ICE
#     plots for the top three descriptors, + interaction heatmap.
# Outputs -> BP_FINAL_MODEL/figures/optimal_descriptor_pipeline/
# =============================================================================
pass  # (Google Colab: mounted Google Drive here)
pass  # (Google Colab: mounted Google Drive here)
import itertools, numpy as np, pandas as pd, matplotlib.pyplot as plt, shap
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

P = "./BP_FINAL_MODEL/descriptors/train_test_split/"
OUT = "./BP_FINAL_MODEL/figures/optimal_descriptor_pipeline/"
LGBM_PARAMS = {"n_estimators": 1784, "learning_rate": 0.06686808497475842, "num_leaves": 18,
    "max_depth": 6, "min_child_samples": 32, "subsample": 0.9921028774475024,
    "colsample_bytree": 0.7660499104221706, "reg_alpha": 0.06008952920081579,
    "reg_lambda": 0.001212337250189692, "random_state": 42, "n_jobs": -1, "verbosity": -1}
N_SAMPLE, N_BG, GRID = 400, 800, 25
rng = np.random.default_rng(42)

tr = pd.read_csv(P + "train_optimal.csv"); te = pd.read_csv(P + "test_optimal.csv")
feats = [c for c in tr.columns if c.startswith(("rdkit2d_", "mordred_"))]; TB = tr.columns[5]
short = [f.replace("rdkit2d_", "").replace("mordred_", "") for f in feats]
sc = StandardScaler().fit(tr[feats].values)
Xtr, Xte = sc.transform(tr[feats].values), sc.transform(te[feats].values)
ytr, yte = tr[TB].values, te[TB].values
model = LGBMRegressor(**LGBM_PARAMS).fit(Xtr, ytr)
print(f"Final model test R2 = {r2_score(yte, model.predict(Xte)):.4f} (published 0.9596)")

# ---- 1. additive (no-interaction) LightGBM ----------------------------------
add = LGBMRegressor(**LGBM_PARAMS, interaction_constraints=[[i] for i in range(len(feats))])
cv = KFold(10, shuffle=True, random_state=42)
res = []
for name, m in [("LightGBM (final, interactions allowed)", LGBMRegressor(**LGBM_PARAMS)),
                ("LightGBM, additive (interactions forbidden)", add)]:
    s = cross_val_score(m, Xtr, ytr, cv=cv, scoring="r2")
    m.fit(Xtr, ytr); p = m.predict(Xte)
    res.append({"Model": name, "CV R2": f"{s.mean():.4f} ± {s.std(ddof=1):.4f}",
                "Test R2": round(r2_score(yte, p), 4), "Test RMSE (K)": round(mean_squared_error(yte, p) ** 0.5, 2)})
res = pd.DataFrame(res); print("\n", res.to_string(index=False)); res.to_csv(OUT + "interaction_additive_comparison.csv", index=False)

# ---- 2. SHAP interaction values ----------------------------------------------
idx = rng.choice(len(Xtr), size=min(1500, len(Xtr)), replace=False)
siv = shap.TreeExplainer(model).shap_interaction_values(Xtr[idx])          # n x p x p
A = np.abs(siv).mean(axis=0)                                                 # p x p mean |phi_ij|
main = np.trace(A); inter = A.sum() - main
print(f"\nSHAP: share of total absolute attribution from pairwise interactions = {100*inter/A.sum():.1f}% "
      f"(main effects {100*main/A.sum():.1f}%)")
Marr = 2 * A                                              # pair strength = phi_ij + phi_ji
np.fill_diagonal(Marr, np.diag(A))                        # diagonal = main effects
M = pd.DataFrame(Marr, index=short, columns=short)
M.to_csv(OUT + "shap_interaction_matrix.csv")
pairs = sorted([(short[i], short[j], 2 * A[i, j]) for i, j in itertools.combinations(range(len(feats)), 2)],
               key=lambda t: -t[2])
print("Strongest pairwise interactions (mean |SHAP interaction|, K):")
for a, b, v in pairs[:6]:
    print(f"  {a:>12s} x {b:<12s} {v:6.2f}")

# ---- 3. Friedman H-statistic for top-3 pairs -----------------------------------
top3 = list(np.argsort(-np.abs(shap.TreeExplainer(model).shap_values(Xtr[idx])).mean(0))[:3])
print("\nTop three descriptors:", [short[i] for i in top3])
S = Xtr[rng.choice(len(Xtr), N_SAMPLE, replace=False)]; BG = Xtr[rng.choice(len(Xtr), N_BG, replace=False)]
def pd_at(cols, vals):                    # partial dependence at each row of vals (n x len(cols))
    out = np.empty(len(vals))
    for r, v in enumerate(vals):
        Z = BG.copy(); Z[:, cols] = v; out[r] = model.predict(Z).mean()
    return out - out.mean()
pd1 = {j: pd_at([j], S[:, [j]]) for j in top3}
H = []
for j, k in itertools.combinations(top3, 2):
    pjk = pd_at([j, k], S[:, [j, k]])
    h2 = ((pjk - pd1[j] - pd1[k]) ** 2).sum() / (pjk ** 2).sum()
    H.append({"Pair": f"{short[j]} x {short[k]}", "H2": round(h2, 4), "H": round(np.sqrt(h2), 3),
              "SHAP interaction (K)": round(2 * A[j, k], 2)})
H = pd.DataFrame(H); print("\nFriedman H-statistic (0 = additive, 1 = pure interaction):\n", H.to_string(index=False))
H.to_csv(OUT + "friedman_H_top3.csv", index=False)

# ---- 4. figure -----------------------------------------------------------------
unsc = lambda j, z: z * sc.scale_[j] + sc.mean_[j]
fig = plt.figure(figsize=(16, 9.5)); gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.1])
for c, (j, k) in enumerate(itertools.combinations(top3, 2)):
    ax = fig.add_subplot(gs[0, c])
    gj = np.linspace(*np.percentile(Xtr[:, j], [2, 98]), GRID); gk = np.linspace(*np.percentile(Xtr[:, k], [2, 98]), GRID)
    GJ, GK = np.meshgrid(gj, gk)
    vals = np.empty(GJ.shape)
    for a in range(GRID):
        for b in range(GRID):
            Zb = BG.copy(); Zb[:, j] = GJ[a, b]; Zb[:, k] = GK[a, b]; vals[a, b] = model.predict(Zb).mean()
    cs = ax.contourf(unsc(j, GJ), unsc(k, GK), vals, levels=14, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="Partial dependence, $T_b$ (K)")
    ax.set_xlabel(short[j]); ax.set_ylabel(short[k])
    ax.set_title(f"({'abc'[c]}) 2D PD, H = {H.iloc[c]['H']:.2f}", loc="left", fontsize=10)
for c, j in enumerate(top3):
    ax = fig.add_subplot(gs[1, c]); grid = np.linspace(*np.percentile(Xtr[:, j], [2, 98]), 30)
    ice_rows = S[:120]
    ice = np.array([[model.predict(np.where(np.arange(len(feats)) == j, g, r)[None, :])[0] for g in grid] for r in ice_rows])
    ice -= ice[:, [0]]
    ax.plot(unsc(j, grid), ice.T, color="#8c8c8c", alpha=0.25, lw=0.7)
    ax.plot(unsc(j, grid), ice.mean(0), color="#D55E00", lw=2.4, label="Average (PD)")
    ax.set_xlabel(short[j]); ax.set_ylabel("Centred prediction, ΔTb (K)")
    ax.set_title(f"({'def'[c]}) Centred ICE: {short[j]}", loc="left", fontsize=10); ax.legend(fontsize=8)
ax = fig.add_subplot(gs[:, 3]); Mo = Marr.copy().astype(float); np.fill_diagonal(Mo, np.nan)
im = ax.imshow(Mo, cmap="magma_r"); fig.colorbar(im, ax=ax, shrink=0.6, label="Mean |SHAP interaction| (K)")
ax.set_xticks(range(len(short))); ax.set_xticklabels(short, rotation=90, fontsize=8)
ax.set_yticks(range(len(short))); ax.set_yticklabels(short, fontsize=8)
ax.set_title(f"(g) Pairwise interaction strength\n({100*inter/A.sum():.0f}% of attribution from interactions)",
             loc="left", fontsize=10)
plt.tight_layout(); plt.savefig(OUT + "fig_interactions_top3.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT + "fig_interactions_top3.pdf", bbox_inches="tight"); plt.show()
print("Saved interaction_additive_comparison.csv, shap_interaction_matrix.csv, friedman_H_top3.csv, figure ->", OUT)
