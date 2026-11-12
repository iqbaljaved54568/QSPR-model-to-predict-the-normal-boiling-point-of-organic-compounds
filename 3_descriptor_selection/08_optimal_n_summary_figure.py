"""
Summary figure of the optimal descriptor count per model

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

fig, ax = plt.subplots(figsize=(10, 6))
colors = [MODEL_COLORS.get(m, "#333") for m in summary_df["model"]]
bars = ax.bar(summary_df["model"], summary_df["optimal_n_99pct"], color=colors, edgecolor="white")
max_height = summary_df["optimal_n_99pct"].max()
label_offset = max_height * 0.03
for bar, n_opt, r2 in zip(bars, summary_df["optimal_n_99pct"], summary_df["test_r2_at_optimal_n_99pct"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + label_offset,
            f"n={n_opt}\nR²={r2:.3f}", ha="center", va="bottom", fontsize=8.5)
ax.set_ylim(0, max_height * 1.25)
ax.set_ylabel("Smallest n Reaching 99% of Peak Test R²", fontsize=10.5)
ax.set_title("Optimal Descriptor Count per Model (99%-of-peak-R² rule)",
             fontsize=12, fontweight="bold")
ax.grid(True, axis="y"); ax.set_axisbelow(True)
plt.tight_layout()
save_fig(fig, f"{OUT_DIR}fig_optimal_n_summary.png")
plt.show(); plt.close(fig)
