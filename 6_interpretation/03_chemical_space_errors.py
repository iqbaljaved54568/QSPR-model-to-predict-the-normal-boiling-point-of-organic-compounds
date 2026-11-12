"""
PCA/t-SNE chemical-space map of errors (Table S14, Figure S10)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# Chemical-space map of the final model's errors (PCA + t-SNE of the ten descriptors)
#
# Residuals are OUT-OF-SAMPLE for all 5,071 compounds:
#   training compounds -> 10-fold cross-validated (out-of-fold) predictions,
#   test compounds     -> predictions of the final model.
# Panels: (a) PCA, (b) t-SNE, both coloured by |residual|; (c) MAE per chemical-space
# region (k-means clusters on the standardised descriptors) with dominant chemical family.
# Statistics: Kruskal-Wallis test of |residual| across regions; MAE range between regions.
# Outputs -> BP_FINAL_MODEL/figures/optimal_descriptor_pipeline/
# =============================================================================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit"], check=True)
pass  # (Google Colab: mounted Google Drive here)
pass  # (Google Colab: mounted Google Drive here)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from rdkit import Chem
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, cross_val_predict
from scipy.stats import kruskal

P = "./BP_FINAL_MODEL/descriptors/train_test_split/"
OUT = "./BP_FINAL_MODEL/figures/optimal_descriptor_pipeline/"
K = 8                      # number of chemical-space regions
LGBM_PARAMS = {"n_estimators": 1784, "learning_rate": 0.06686808497475842, "num_leaves": 18,
    "max_depth": 6, "min_child_samples": 32, "subsample": 0.9921028774475024,
    "colsample_bytree": 0.7660499104221706, "reg_alpha": 0.06008952920081579,
    "reg_lambda": 0.001212337250189692, "random_state": 42, "n_jobs": -1, "verbosity": -1}

tr = pd.read_csv(P + "train_optimal.csv"); te = pd.read_csv(P + "test_optimal.csv")
feats = [c for c in tr.columns if c.startswith(("rdkit2d_", "mordred_"))]; TB = tr.columns[5]
sc = StandardScaler().fit(tr[feats].values)
Xtr, Xte = sc.transform(tr[feats].values), sc.transform(te[feats].values)
oof = cross_val_predict(LGBMRegressor(**LGBM_PARAMS), Xtr, tr[TB].values, cv=KFold(10, shuffle=True, random_state=42))
pte = LGBMRegressor(**LGBM_PARAMS).fit(Xtr, tr[TB].values).predict(Xte)
d = pd.concat([tr[["Name", "smiles", TB]].assign(Set="train", Pred=oof),
               te[["Name", "smiles", TB]].assign(Set="test", Pred=pte)], ignore_index=True)
d["Residual"] = d["Pred"] - d[TB]; d["AbsRes"] = d["Residual"].abs()
X = np.vstack([Xtr, Xte])
print(f"Out-of-sample MAE: training (10-fold OOF) {d[d.Set=='train'].AbsRes.mean():.2f} K, "
      f"test {d[d.Set=='test'].AbsRes.mean():.2f} K")

# chemical family (same priority rule as the revised Figure 1)
FAM = ["Phosphorus", "Heteroaromatic", "Sulfur", "Nitrogen", "Halogenated", "Oxygenated",
       "Aromatic hydrocarbon", "Aliphatic hydrocarbon"]
def family(s):
    m = Chem.MolFromSmiles(s); el = {a.GetSymbol() for a in m.GetAtoms()}
    het_ar = any(a.GetIsAromatic() and a.GetSymbol() != "C" for a in m.GetAtoms())
    ar = any(a.GetIsAromatic() for a in m.GetAtoms())
    flags = ["P" in el, het_ar, "S" in el, "N" in el, bool(el & {"Cl", "Br", "I", "F"}), "O" in el,
             el <= {"C", "H"} and ar, el <= {"C", "H"} and not ar]
    return next((f for f, b in zip(FAM, flags) if b), "Other")
d["Family"] = d["smiles"].apply(family)

# embeddings and regions
pca = PCA(n_components=2, random_state=42).fit(Xtr); Zp = pca.transform(X)
ev = pca.explained_variance_ratio_ * 100
Zt = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(X)
d["Region"] = KMeans(n_clusters=K, n_init=20, random_state=42).fit_predict(X) + 1

reg = d.groupby("Region").agg(n=("AbsRes", "size"), MAE=("AbsRes", "mean"),
                               RMSE=("Residual", lambda r: np.sqrt(np.mean(r ** 2))),
                               Median_abs=("AbsRes", "median"), Bias=("Residual", "mean"),
                               Pct_over_57K=("AbsRes", lambda r: 100 * np.mean(r > 57.1)),
                               Mean_Tb=(TB, "mean"))
reg["Dominant family (%)"] = d.groupby("Region")["Family"].agg(
    lambda f: f"{f.value_counts().index[0]} ({100 * f.value_counts(normalize=True).iloc[0]:.0f}%)")
reg = reg.sort_values("MAE"); print("\n", reg.round(2).to_string())
H, pk = kruskal(*[g["AbsRes"].values for _, g in d.groupby("Region")])
print(f"\nKruskal-Wallis test of |residual| across {K} regions: H = {H:.1f}, p = {pk:.2g}")
print(f"MAE ranges from {reg.MAE.min():.1f} to {reg.MAE.max():.1f} K "
      f"({reg.MAE.max() / reg.MAE.min():.1f}-fold); overall {d.AbsRes.mean():.1f} K")
print("\nMAE by chemical family:\n", d.groupby("Family")["AbsRes"].agg(["size", "mean"]).round(2)
      .sort_values("mean").to_string())
reg.round(3).to_csv(OUT + "chemical_space_region_errors.csv")
d.assign(PC1=Zp[:, 0], PC2=Zp[:, 1], tSNE1=Zt[:, 0], tSNE2=Zt[:, 1]).to_csv(
    OUT + "chemical_space_coordinates_residuals.csv", index=False)

# figure
vmax = np.percentile(d.AbsRes, 97); norm = Normalize(0, vmax); o = d.AbsRes.argsort().values
fig = plt.figure(figsize=(17, 5.6)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9])
for c, (Z, t, xl, yl) in enumerate([(Zp, "(a) PCA", f"PC1 ({ev[0]:.1f}%)", f"PC2 ({ev[1]:.1f}%)"),
                                    (Zt, "(b) t-SNE", "t-SNE 1", "t-SNE 2")]):
    ax = fig.add_subplot(gs[0, c])
    s = ax.scatter(Z[o, 0], Z[o, 1], c=d.AbsRes.values[o], cmap="magma_r", norm=norm,
                   s=np.where(d.AbsRes.values[o] > 57.1, 22, 6), alpha=0.85, edgecolors="none", rasterized=True)
    if c == 1:
        for r, g in d.groupby("Region"):
            cx, cy = np.median(Zt[g.index, 0]), np.median(Zt[g.index, 1])
            ax.text(cx, cy, str(r), fontsize=11, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="#333", alpha=0.85))
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(t + ", coloured by |residual|", loc="left", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
cb = fig.colorbar(s, ax=fig.axes[:2], shrink=0.85, pad=0.01, extend="max"); cb.set_label("|Predicted − experimental $T_b$| (K)")
ax = fig.add_subplot(gs[0, 2]); y = np.arange(len(reg))
ax.barh(y, reg.MAE, color=plt.cm.magma_r(norm(reg.MAE.values)), edgecolor="#333", lw=0.5)
ax.axvline(d.AbsRes.mean(), color="#333", ls="--", lw=1, label=f"Overall MAE {d.AbsRes.mean():.1f} K")
ax.set_yticks(y); ax.set_yticklabels([f"Region {r} (n={n})\n{f}" for r, n, f in
                                       zip(reg.index, reg.n, reg["Dominant family (%)"])], fontsize=7.5)
ax.set_xlabel("MAE (K)"); ax.legend(fontsize=8, loc="lower right")
ax.set_title(f"(c) Error by region (Kruskal–Wallis p = {pk:.1g})", loc="left", fontsize=10.5)
ax.spines[["top", "right"]].set_visible(False)
plt.savefig(OUT + "fig_chemical_space_residuals.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT + "fig_chemical_space_residuals.pdf", bbox_inches="tight"); plt.show()
print("Saved figure, region table and coordinates to", OUT)
