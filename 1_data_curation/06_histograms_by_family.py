"""
Molecular mass and Tb histograms coloured by chemical family (Figure 2, Figure S1)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# Figure 2: molecular-mass and Tb histograms colour-coded by chemical family
#
# Each compound gets ONE family (the eight named in the Data section), by priority:
#   Phosphorus > Heteroaromatic > Sulfur > Nitrogen > Halogenated (Cl/Br/I) >
#   Oxygenated > Aromatic hydrocarbon > Aliphatic hydrocarbon
# i.e. a compound is assigned to the first family it belongs to in this list, so the
# rarer heteroatom families stay visible. (Multi-membership counts are also reported.)
#
# Outputs (BP_FINAL_MODEL/figures/): fig1_histograms_by_family.png/.pdf,
#   family_counts.csv, family_tails.csv (composition of the 5% tails), table_SI_family_tails.tex
# =============================================================================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit"], check=True)
pass  # (Google Colab: mounted Google Drive here)
pass  # (Google Colab: mounted Google Drive here)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from rdkit import Chem

IN = "./BP_FINAL_MODEL/Final_clean-organic.csv"
OUT = "./BP_FINAL_MODEL/figures/"
TB, MW = "boiling point (K)", "molecular weight"

FAMILIES = ["Phosphorus", "Heteroaromatic", "Sulfur", "Nitrogen", "Halogenated",
            "Oxygenated", "Aromatic hydrocarbon", "Aliphatic hydrocarbon"]
COLORS = {"Aliphatic hydrocarbon": "#999999", "Aromatic hydrocarbon": "#E69F00",
          "Oxygenated": "#56B4E9", "Halogenated": "#009E73", "Nitrogen": "#0072B2",
          "Sulfur": "#F0E442", "Heteroaromatic": "#D55E00", "Phosphorus": "#CC79A7"}  # Okabe-Ito

def memberships(smi):
    m = Chem.MolFromSmiles(smi)
    el = {a.GetSymbol() for a in m.GetAtoms()}
    arom_c = any(a.GetIsAromatic() and a.GetSymbol() == "C" for a in m.GetAtoms())
    return {"Phosphorus": "P" in el,
            "Heteroaromatic": any(a.GetIsAromatic() and a.GetSymbol() != "C" for a in m.GetAtoms()),
            "Sulfur": "S" in el, "Nitrogen": "N" in el,
            "Halogenated": bool(el & {"Cl", "Br", "I", "F"}), "Oxygenated": "O" in el,
            "Aromatic hydrocarbon": el <= {"C", "H"} and arom_c,
            "Aliphatic hydrocarbon": el <= {"C", "H"} and not arom_c}

df = pd.read_csv(IN)
mem = pd.DataFrame([memberships(s) for s in df["smiles"]])[FAMILIES]
df["Family"] = mem.apply(lambda r: next((f for f in FAMILIES if r[f]), "Unassigned"), axis=1)  # first in priority
print(f"Compounds: {len(df)}; unassigned: {(df['Family'] == 'Unassigned').sum()}")

counts = pd.DataFrame({"Assigned (one family each)": df["Family"].value_counts(),
                       "Contains this family (any)": mem.sum()}).reindex(FAMILIES).fillna(0).astype(int)
counts["Assigned %"] = (100 * counts.iloc[:, 0] / len(df)).round(1)
print("\n", counts.to_string()); counts.to_csv(OUT + "family_counts.csv")

# ---- tails: lowest/highest 5% of Tb and of MW ----------------------------------
rows = {}
for var, lab in [(TB, "Tb"), (MW, "MW")]:
    lo, hi = df[var].quantile([0.05, 0.95])
    for name, mask in [(f"{lab} lowest 5% (< {lo:.0f})", df[var] < lo),
                       (f"{lab} middle 90%", df[var].between(lo, hi)),
                       (f"{lab} highest 5% (> {hi:.0f})", df[var] > hi)]:
        rows[name] = (100 * df.loc[mask, "Family"].value_counts(normalize=True)).reindex(FAMILIES).fillna(0)
        rows[name]["n"] = int(mask.sum())
tails = pd.DataFrame(rows).T
print("\nFamily composition (%) of the distribution tails:\n", tails.round(1).to_string())
tails.round(1).to_csv(OUT + "family_tails.csv")

# ---- figure ----------------------------------------------------------------------
plot_order = FAMILIES[::-1]                       # aliphatic at the bottom of the stack
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), gridspec_kw={"height_ratios": [1.6, 1]})
for col, (var, xlab, width) in enumerate([(MW, "Molecular mass (g mol$^{-1}$)", 10), (TB, "$T_b$ (K)", 10)]):
    edges = np.arange(np.floor(df[var].min() / width) * width, df[var].max() + width, width)
    data = [df.loc[df["Family"] == f, var].values for f in plot_order]
    ax = axes[0, col]
    ax.hist(data, bins=edges, stacked=True, color=[COLORS[f] for f in plot_order],
            edgecolor="white", linewidth=0.3, label=plot_order)
    ax.set_ylabel("Number of compounds"); ax.set_xlabel(xlab)
    ax.set_title(f"({'ab'[col]}) {'Molecular mass' if col == 0 else 'Boiling point'}", loc="left", fontweight="bold")
    for q in df[var].quantile([0.05, 0.95]):
        ax.axvline(q, color="#333", ls=":", lw=1)
    # composition per bin (100% stacked), coarser bins
    edges2 = np.linspace(df[var].min(), df[var].max(), 16)
    b = pd.cut(df[var], edges2, include_lowest=True)
    comp = pd.crosstab(b, df["Family"], normalize="index").reindex(columns=plot_order).fillna(0)
    n_bin = pd.crosstab(b, df["Family"]).sum(axis=1)
    mids = [iv.mid for iv in comp.index]; bottom = np.zeros(len(comp))
    ax2 = axes[1, col]
    for f in plot_order:
        ax2.bar(mids, comp[f].values * 100, width=(edges2[1] - edges2[0]) * 0.95, bottom=bottom,
                color=COLORS[f], edgecolor="white", linewidth=0.3)
        bottom += comp[f].values * 100
    for x, n in zip(mids, n_bin.values):
        ax2.text(x, 101, str(n), ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax2.set_ylim(0, 112); ax2.set_ylabel("Share of bin (%)"); ax2.set_xlabel(xlab)
    ax2.set_title(f"({'cd'[col]}) Family composition per bin (n above bars)", loc="left", fontweight="bold")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles[::-1], labels[::-1], loc="upper center", ncol=4, frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, 1.03))
for a in axes.flat:
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.savefig(OUT + "fig1_histograms_by_family.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT + "fig1_histograms_by_family.pdf", bbox_inches="tight")
plt.show()

# ---- LaTeX table of tail composition ---------------------------------------------
cols = FAMILIES
tex = ("\\begin{table}[htbp]\n\\centering\n\\caption{Chemical-family composition (\\%) of the lowest and highest "
       "5\\% of the molecular-mass and boiling-point distributions, compared with the central 90\\%. Each compound "
       "is assigned to one family by the priority P $>$ heteroaromatic $>$ S $>$ N $>$ halogen $>$ O $>$ aromatic "
       "hydrocarbon $>$ aliphatic hydrocarbon.}\n\\label{tab:family_tails}\n\\scriptsize\n\\begin{tabular}{l" +
       "r" * (len(cols) + 1) + "}\n\\toprule\nSubset & $n$ & " + " & ".join(c.replace(" hydrocarbon", " HC") for c in cols) +
       " \\\\\n\\midrule\n" +
       "\n".join(f"{i.replace('%', chr(92)+'%').replace('<', '$<$').replace('>', '$>$')} & {int(r['n'])} & " +
                 " & ".join(f"{r[c]:.1f}" for c in cols) + " \\\\" for i, r in tails.iterrows()) +
       "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
open(OUT + "table_SI_family_tails.tex", "w").write(tex)
print("Saved figure, family_counts.csv, family_tails.csv, table_SI_family_tails.tex to", OUT)
