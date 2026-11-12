"""
Boiling-point strata and train/test counts per bin

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# Stratified split: recover the boiling-point strata and tabulate train/test per bin
# Uses train_optimal.csv / test_optimal.csv.
#
# 1. Tests which binning reproduces the split: for a split stratified on a binning,
#    every bin holds ~20% test compounds (deviation <= ~1 compound per bin).
#    Tries quantile bins (pd.qcut, k = 2..20) and equal-width bins (pd.cut, k = 2..20).
# 2. For the best-matching scheme (or the one you set in USE_SCHEME), writes the
#    bin edges and train/test counts per bin, plus a train-vs-test distribution test.
# =============================================================================
pass  # (Google Colab: mounted Google Drive here)
pass  # (Google Colab: mounted Google Drive here)
import numpy as np, pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

P = "./BP_FINAL_MODEL/descriptors/train_test_split/"
OUT = "./BP_FINAL_MODEL/figures/"
USE_SCHEME = None          # e.g. ("qcut", 10) to force a scheme once you know the original one

tr = pd.read_csv(P + "train_optimal.csv"); te = pd.read_csv(P + "test_optimal.csv")
TB = tr.columns[5]
d = pd.concat([tr[[TB]].assign(Set="Train"), te[[TB]].assign(Set="Test")], ignore_index=True)
frac = (d["Set"] == "Test").mean()
print(f"Train {len(tr)}, test {len(te)}, test fraction {frac:.4f}")

def table(kind, k):
    b = (pd.qcut(d[TB], k, duplicates="drop") if kind == "qcut" else pd.cut(d[TB], k))
    t = pd.crosstab(b, d["Set"])[["Train", "Test"]]
    t["Total"] = t.sum(axis=1)
    t["Test_%"] = 100 * t["Test"] / t["Total"]
    t["Dev_from_expected"] = t["Test"] - t["Total"] * frac       # in compounds
    return t

scores = []
for kind in ["qcut", "cut"]:
    for k in range(2, 21):
        t = table(kind, k)
        if (t["Total"] == 0).any():
            continue
        scores.append({"Scheme": kind, "k": k, "max_|dev|_compounds": t["Dev_from_expected"].abs().max(),
                       "min_bin_n": int(t["Total"].min())})
scores = pd.DataFrame(scores).sort_values("max_|dev|_compounds")
print("\nHow well each binning matches an exact 80/20 split within every bin "
      "(a stratifying scheme should show max |dev| of about 1 compound or less):")
print(scores.head(12).round(2).to_string(index=False))

kind, k = USE_SCHEME or (scores.iloc[0]["Scheme"], int(scores.iloc[0]["k"]))
t = table(kind, k)
t.index = [f"{iv.left:.1f}-{iv.right:.1f}" for iv in t.index]
t.index.name = "Tb bin (K)"
t.loc["Total"] = [t["Train"].sum(), t["Test"].sum(), t["Total"].sum(),
                  100 * t["Test"].sum() / t["Total"].sum(), np.nan]
print(f"\nSelected scheme: {kind}, k = {k}")
print(t.round(1).to_string())

chi = chi2_contingency(t.iloc[:-1][["Train", "Test"]].values)
ks = ks_2samp(tr[TB], te[TB])
print(f"\nTrain vs test: chi-square across bins p = {chi.pvalue:.3f}; "
      f"Kolmogorov-Smirnov p = {ks.pvalue:.3f}")
print(f"Range: train {tr[TB].min():.2f}-{tr[TB].max():.2f} K, test {te[TB].min():.2f}-{te[TB].max():.2f} K")

t.to_csv(OUT + f"split_strata_{kind}{k}.csv")
rows = "\n".join(f"{i} & {int(r.Train)} & {int(r.Test)} & {int(r.Total)} & {r['Test_%']:.1f} \\\\"
                 for i, r in t.iloc[:-1].iterrows())
tot = t.loc["Total"]
tex = (r"""\begin{table}[htbp]
\centering
\caption{Boiling-point strata used for the stratified 80/20 train/test split (""" +
       ("equal-frequency" if kind == "qcut" else "equal-width") + f""" bins, $k = {k}$), with the number of
compounds per stratum in the training and test sets. Train vs.\\ test distribution: $\\chi^2$ test $p = {chi.pvalue:.2f}$;
Kolmogorov--Smirnov $p = {ks.pvalue:.2f}$.}}
\\label{{tab:split_strata}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
$T_b$ stratum (K) & Training & Test & Total & Test (\\%) \\\\
\\midrule
""" + rows + f"""
\\midrule
Total & {int(tot.Train)} & {int(tot.Test)} & {int(tot.Total)} & {tot['Test_%']:.1f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")
open(OUT + f"table_SI_split_strata_{kind}{k}.tex", "w").write(tex)
print("\nSaved split_strata CSV and LaTeX table to", OUT)
