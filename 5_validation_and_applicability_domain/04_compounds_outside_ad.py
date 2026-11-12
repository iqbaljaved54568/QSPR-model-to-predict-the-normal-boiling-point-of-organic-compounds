"""
Test compounds outside the applicability domain (Table S21)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# SI table: the 49 test compounds outside the applicability domain
# Reads the file the final pipeline already saved -- nothing is retrained.
# Prints a check summary and writes a LaTeX longtable.
# =============================================================================
pass  # (Google Colab: mounted Google Drive here)
pass  # (Google Colab: mounted Google Drive here)
import pandas as pd

DIR = "./BP_FINAL_MODEL/figures/optimal_descriptor_pipeline/"
H_STAR, S = 0.00814, 19.023

d = pd.read_csv(DIR + "outliers_outside_AD_optimal.csv")
print("Columns:", list(d.columns))
print(f"\nCompounds outside AD: {len(d)}  ({100*len(d)/1015:.2f}% of 1,015)")
print(d["Outlier_Type"].value_counts().to_string())
print(f"Max leverage: {d['Leverage_h'].max():.5f} | max |std. residual|: {d['Std_Residual'].abs().max():.2f}")

name_col = "Name" if "Name" in d.columns else d.columns[0]
smi_col = next(c for c in d.columns if c.lower() in ("smiles", "canonical_smiles"))
type_short = d["Outlier_Type"].str[0]                      # A / B / C
order = {"C": 0, "B": 1, "A": 2}
d = d.assign(_t=type_short.map(order)).sort_values(["_t", "Abs_Error_K"], ascending=[True, False])

def esc(x):
    return (str(x).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("#", r"\#")
                  .replace("%", r"\%").replace("&", r"\&").replace("$", r"\$")
                  .replace("{", r"\{").replace("}", r"\}"))

rows = []
for i, r in enumerate(d.itertuples(index=False), 1):
    rr = r._asdict()
    rows.append(f"{i} & {esc(rr[name_col]).title()} & {esc(rr['CAS_No_'] if 'CAS_No_' in rr else '')} & "
                f"\\texttt{{{esc(rr[smi_col])}}} & {rr['y_actual']:.1f} & {rr['y_predicted']:.1f} & "
                f"{rr['Leverage_h']:.4f} & {rr['Std_Residual']:+.2f} & {rr['Outlier_Type'][0]} \\\\")

tex = r"""\begin{longtable}{rlllrrrrc}
\caption{Test-set compounds outside the applicability domain of the final LightGBM model
($n = 49$, 4.83\% of 1,015). Type A: $h_i > h^* = 0.00814$ only; Type B: $|\hat{\sigma}_i| > 3$ only
($|y_i - \hat{y}_i| > 57.1$ K); Type C: both. $\hat{\sigma}_i = (y_i - \hat{y}_i)/s$ with $s = 19.023$ K.}
\label{tab:ad_outside}\\
\toprule
\# & Compound & CAS No. & SMILES & $T_b$ (K) & Pred. (K) & $h_i$ & $\hat{\sigma}_i$ & Type \\
\midrule
\endfirsthead
\toprule
\# & Compound & CAS No. & SMILES & $T_b$ (K) & Pred. (K) & $h_i$ & $\hat{\sigma}_i$ & Type \\
\midrule
\endhead
\bottomrule
\endfoot
""" + "\n".join(rows) + "\n\\end{longtable}\n"

# CAS column has a space/dot in its name; fill it properly
cas_col = next((c for c in d.columns if "cas" in c.lower()), None)
if cas_col:
    tex_rows = []
    for i, (_, r) in enumerate(d.iterrows(), 1):
        tex_rows.append(f"{i} & {esc(r[name_col]).title()} & {esc(r[cas_col])} & "
                        f"\\texttt{{{esc(r[smi_col])}}} & {r['y_actual']:.1f} & {r['y_predicted']:.1f} & "
                        f"{r['Leverage_h']:.4f} & {r['Std_Residual']:+.2f} & {r['Outlier_Type'][0]} \\\\")
    tex = tex.split("\\endfoot\n")[0] + "\\endfoot\n" + "\n".join(tex_rows) + "\n\\end{longtable}\n"

open(DIR + "table_SI_outside_AD.tex", "w").write(tex)
d.drop(columns="_t").to_csv(DIR + "table_SI_outside_AD.csv", index=False)
print("\nSaved -> table_SI_outside_AD.tex and table_SI_outside_AD.csv in", DIR)
print("\n--- first rows of the LaTeX table ---")
print("\n".join(tex.splitlines()[:22]))
