"""
Y-randomisation extended to 1,000 permutations (Table S7, Figure S6)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# Y-randomisation: full distribution, extended to N_TOTAL permutations
# Reuses the saved first 100 scrambled runs
# (yrand_checkpoint_LightGBM.npz, seeds 42-141) and adds new ones with the SAME
# procedure and the next seeds, so the first 100 are identical to the published ones.
#
# Procedure per permutation (identical to the pipeline): shuffle Tb of the training
# set; fit LightGBM (final hyperparameters) on 85% of it with early stopping on the
# other 15% (30 rounds); record R2 on that 15% ("CV-like") and on the real test set.
#
# Outputs (optimal_descriptor_pipeline folder): yrand_values_<N>.csv (every value),
# fig_yrand_distribution.png, table_SI_yrand.tex, and printed statistics.
# =============================================================================
pass  # (Google Colab: mounted Google Drive here)
pass  # (Google Colab: mounted Google Drive here)
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

N_TOTAL = 1000
P = "./BP_FINAL_MODEL/descriptors/train_test_split/"
OUT = "./BP_FINAL_MODEL/figures/optimal_descriptor_pipeline/"
REAL_CV, REAL_TEST = 0.9568, 0.9596          # published 10-fold CV R2 and test R2
LGBM_PARAMS = {"n_estimators": 1784, "learning_rate": 0.06686808497475842, "num_leaves": 18,
    "max_depth": 6, "min_child_samples": 32, "subsample": 0.9921028774475024,
    "colsample_bytree": 0.7660499104221706, "reg_alpha": 0.06008952920081579,
    "reg_lambda": 0.001212337250189692, "random_state": 42, "n_jobs": -1, "verbosity": -1}

tr = pd.read_csv(P + "train_optimal.csv"); te = pd.read_csv(P + "test_optimal.csv")
feats = [c for c in tr.columns if c.startswith(("rdkit2d_", "mordred_"))]; TB = tr.columns[5]
sc = StandardScaler().fit(tr[feats].values)
Xtr, Xte = sc.transform(tr[feats].values), sc.transform(te[feats].values)
ytr, yte = tr[TB].values, te[TB].values

def one(seed):                                   # identical to _yrand_lgbm in the pipeline
    rng = np.random.RandomState(seed)
    ys = rng.permutation(ytr)
    idx = rng.permutation(len(ytr)); nv = max(10, int(0.15 * len(ytr)))
    v, f = idx[:nv], idx[nv:]
    m = LGBMRegressor(**{**LGBM_PARAMS, "random_state": seed, "n_jobs": 1})
    m.fit(Xtr[f], ys[f], eval_set=[(Xtr[v], ys[v])],
          callbacks=[early_stopping(30, verbose=False), log_evaluation(0)])
    return r2_score(ys[v], m.predict(Xtr[v])), r2_score(yte, m.predict(Xte))

ck = np.load(OUT + "yrand_checkpoint_LightGBM.npz")
cv0, te0 = list(ck["cv_rand"][:100]), list(ck["test_rand"][:100])
chk = one(42)
print(f"Reproduction check, seed 42: new ({chk[0]:.5f}, {chk[1]:.5f}) vs saved ({cv0[0]:.5f}, {te0[0]:.5f})")
new = joblib.Parallel(n_jobs=-1)(joblib.delayed(one)(42 + i) for i in range(100, N_TOTAL))
cv = np.array(cv0 + [a for a, _ in new]); tst = np.array(te0 + [b for _, b in new])

def stats(real, r, name):
    n = len(r); k = int((r >= real).sum())
    s = {"Split": name, "N permutations": n, "Real R2": real, "Mean": r.mean(), "SD": r.std(ddof=1),
         "Min": r.min(), "P5": np.percentile(r, 5), "Median": np.median(r), "P95": np.percentile(r, 95),
         "Max": r.max(), "N >= real": k, "Empirical p": (k + 1) / (n + 1),
         "Real - mean (SD units)": (real - r.mean()) / r.std(ddof=1)}
    return s
res = pd.DataFrame([stats(REAL_CV, cv, "CV-like (85/15 split)"), stats(REAL_TEST, tst, "Test set")])
pd.set_option("display.width", 250)
print("\n", res.round(4).to_string(index=False))
for n in (100, N_TOTAL):
    print(f"Empirical p with {n} permutations (none reaching real R2): {1/(n+1):.4g}")

pd.DataFrame({"seed": 42 + np.arange(len(cv)), "R2_CV_like_scrambled": cv, "R2_test_scrambled": tst}) \
  .to_csv(OUT + f"yrand_values_{len(cv)}.csv", index=False)
res.to_csv(OUT + f"yrand_statistics_{len(cv)}.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, r, real, t in [(axes[0], cv, REAL_CV, "(a) CV-like $R^2$ (85/15 split of training set)"),
                       (axes[1], tst, REAL_TEST, "(b) Test-set $R^2$ (n = 1,015)")]:
    ax.hist(r, bins=40, color="#8c8c8c", edgecolor="white", label=f"Scrambled (n = {len(r)})")
    ax.axvline(real, color="#c0392b", lw=2.2, label=f"Real model, $R^2$ = {real:.4f}")
    ax.set_xlim(min(r.min(), 0) - 0.05, 1.02); ax.set_xlabel("$R^2$"); ax.set_ylabel("Count")
    ax.set_title(t, loc="left", fontsize=10.5); ax.legend(fontsize=8.5, loc="upper center")
    ins = ax.inset_axes([0.08, 0.35, 0.38, 0.38])      # zoom on the scrambled distribution
    ins.hist(r, bins=30, color="#8c8c8c", edgecolor="white"); ins.tick_params(labelsize=7)
    ins.set_title("scrambled (zoom)", fontsize=7.5)
plt.tight_layout(); plt.savefig(OUT + "fig_yrand_distribution.png", dpi=300, bbox_inches="tight"); plt.show()

r = res.set_index("Split")
tex = "\\begin{table}[htbp]\n\\centering\n\\caption{Y-randomisation of the final LightGBM model (" + \
      f"{len(cv)}" + " permutations of the training-set $T_b$; hyperparameters fixed). Empirical $p = (k+1)/(N+1)$, " \
      "where $k$ is the number of permutations with $R^2 \\geq$ the real model.}\n\\label{tab:yrand}\n\\small\n" \
      "\\begin{tabular}{lcc}\n\\toprule\n & CV-like (85/15 split) & Test set \\\\\n\\midrule\n" + \
      "\n".join(f"{lab} & {r.iloc[0][c]:{fmt}} & {r.iloc[1][c]:{fmt}} \\\\" for lab, c, fmt in [
          ("Real-model $R^2$", "Real R2", ".4f"), ("Scrambled mean $R^2$", "Mean", ".4f"),
          ("Scrambled SD", "SD", ".4f"), ("Scrambled min / max", "Min", ".4f"),
          ("Scrambled 95th percentile", "P95", ".4f"), ("Permutations with $R^2 \\geq$ real", "N >= real", ".0f"),
          ("Empirical $p$", "Empirical p", ".2g"), ("Real $-$ mean (in SD units)", "Real - mean (SD units)", ".0f")]) + \
      "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
tex = tex.replace(f"{r.iloc[0]['Min']:.4f} & {r.iloc[1]['Min']:.4f}",
                  f"{r.iloc[0]['Min']:.4f} / {r.iloc[0]['Max']:.4f} & {r.iloc[1]['Min']:.4f} / {r.iloc[1]['Max']:.4f}")
open(OUT + "table_SI_yrand.tex", "w").write(tex)
print("\nSaved yrand_values, yrand_statistics, fig_yrand_distribution.png, table_SI_yrand.tex to", OUT)
