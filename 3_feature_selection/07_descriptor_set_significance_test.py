# Approximate significance test using ONLY what's already saved to disk
# (backward_elimination_results_35desc.csv) -- no need to rerun the sweep.
#
# CAVEAT: this uses an unpaired (Welch's) t-test from summary statistics
# (mean, std, n_folds), not the exact paired test the original code ran
# in-session. Because both n=10 and n=35 were evaluated on the SAME fold
# splits (same random_state), a proper paired test typically has MORE power
# to detect a real difference than this unpaired approximation -- so this
# test is conservative (biased toward "not significant"), not liberal.
# If this test finds a significant difference, you can trust it. If it
# doesn't, a true paired test could theoretically still find significance,
# but this remains a defensible, honestly-caveated number to report.
# =============================================================================

import pandas as pd
from scipy.stats import ttest_ind_from_stats

RESULTS_CSV = "/content/drive/MyDrive/BP_FINAL_MODEL/figures/backward_elimination_35desc/backward_elimination_results_35desc.csv"
CV_SPLITS_USED = 5   # matches CV_SPLITS_RFE in the original sweep

results_df = pd.read_csv(RESULTS_CSV)
lgbm = results_df[results_df["model"] == "LightGBM"].set_index("n_features")

print("Available LightGBM n_features checkpoints:", sorted(lgbm.index.tolist()))

n_a, n_b = 35, 10
if n_a not in lgbm.index or n_b not in lgbm.index:
    raise ValueError(f"n={n_a} or n={n_b} not found in the saved results -- check the list above.")

row_a, row_b = lgbm.loc[n_a], lgbm.loc[n_b]

t_stat, p_value = ttest_ind_from_stats(
    mean1=row_a["cv_r2_mean"], std1=row_a["cv_r2_std"], nobs1=CV_SPLITS_USED,
    mean2=row_b["cv_r2_mean"], std2=row_b["cv_r2_std"], nobs2=CV_SPLITS_USED,
    equal_var=False,  # Welch's -- doesn't assume equal variance between the two steps
)

print(f"\n{'='*64}")
print(f"LightGBM: n={n_a} vs n={n_b} -- approximate (unpaired, Welch's) t-test")
print(f"{'='*64}")
print(f"  n={n_a}  CV R2 = {row_a['cv_r2_mean']:.4f} +/- {row_a['cv_r2_std']:.4f}   Test R2 = {row_a['test_r2']:.4f}")
print(f"  n={n_b}  CV R2 = {row_b['cv_r2_mean']:.4f} +/- {row_b['cv_r2_std']:.4f}   Test R2 = {row_b['test_r2']:.4f}")
print(f"  Difference (CV R2, {n_a} - {n_b}) : {row_a['cv_r2_mean'] - row_b['cv_r2_mean']:.4f}")
print(f"  t-statistic : {t_stat:.4f}")
print(f"  p-value     : {p_value:.4f}   (approximate -- unpaired test, see caveat above)")
print()

if p_value < 0.05:
    print(f"  => Approximately significant at alpha=0.05. Since this unpaired test is")
    print(f"     CONSERVATIVE relative to the true paired design, a significant result")
    print(f"     here is trustworthy -- the {n_a}-descriptor model likely IS genuinely")
    print(f"     better, even though the practical gap is small ({row_a['cv_r2_mean'] - row_b['cv_r2_mean']:.4f} R^2).")
else:
    print(f"  => NOT significant at alpha=0.05 (approximate test).")
    print(f"     No evidence the {n_a}-descriptor model outperforms the {n_b}-descriptor")
    print(f"     model beyond fold-to-fold noise. This supports using the smaller set.")
    print(f"     (Note: a true paired test could in principle be more sensitive and find")
    print(f"     significance where this approximation doesn't -- but absent that, this")
    print(f"     is the most defensible number available without rerunning the sweep.)")

sig_phrase = ("a statistically significant (though practically small) difference"
              if p_value < 0.05 else
              "a difference not statistically distinguishable from CV-fold noise "
              "(approximate unpaired test; the true paired comparison was not recoverable)")
print(f"\nSuggested sentence for the paper:")
print(f'  "Reducing the descriptor set from {n_a} to {n_b} changed LightGBM\'s '
      f'cross-validated R^2 from {row_a["cv_r2_mean"]:.4f} to {row_b["cv_r2_mean"]:.4f} '
      f'(Welch\'s t-test, p={p_value:.3f}), {sig_phrase}."')

print(f"\nAlso available for context:")
print(f"  Test R2 difference ({n_a} - {n_b}): {row_a['test_r2'] - row_b['test_r2']:.4f}")
print(f"  Test RMSE ({n_a} desc): {row_a['test_rmse']:.3f} K   Test RMSE ({n_b} desc): {row_b['test_rmse']:.3f} K")
print(f"  Test MAE  ({n_a} desc): {row_a['test_mae']:.3f} K   Test MAE  ({n_b} desc): {row_b['test_mae']:.3f} K")

# =============================================================================
