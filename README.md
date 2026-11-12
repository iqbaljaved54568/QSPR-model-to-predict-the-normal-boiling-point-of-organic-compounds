# Ten-descriptor LightGBM model for normal boiling points

Code, data and analysis outputs for:

> Javed, M. I.; Waters, L.; Belton, D.; Cooke, D. *A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of Structurally Diverse Organic Compounds: Validation, Applicability Domain and Online Deployment.* Submitted to *ACS Omega*, 2026.

- **Archive (this repository):** https://doi.org/10.5281/zenodo.22311330
- **Web application:** https://bppredictionapp-eghwsz9hbquq3kcrcgjfk5.streamlit.app/ (source: https://github.com/iqbaljaved54568/Bp_prediction_app, archived at https://doi.org/10.5281/zenodo.22311227)

## Model in brief

| | |
|---|---|
| Data | 5,071 organic compounds (C, H, N, O, P, S, Cl, Br, I); 4,144 NIST- and 927 DIPPR-sourced boiling points; 238.65–743.0 K |
| Split | 4,056 training / 1,015 test (boiling-point stratified) |
| Descriptors | 10 two-dimensional descriptors (RDKit Ipc + 9 Mordred), selected from 2,054 |
| Model | LightGBM (Optuna-tuned) |
| Test set | R² = 0.9596, MAE = 12.32 K, RMSE = 18.39 K; 10-fold CV R² = 0.9568 ± 0.0071 |
| Applicability domain | Leverage, h* = 3(p + 1)/n = 0.00814; 96.6% of test compounds inside |

## Repository contents

```
1_data_curation/                        Data selection, curation, distributions, functional groups
2_descriptor_calculation/               RDKit and Mordred descriptors (2,054)
3_descriptor_selection/                 Filters (-> 94), RF ranking (-> 35), train/test split,
                                        backward elimination (-> 10), robustness checks
4_model_development/                    Optuna tuning, five models, FINAL 10-descriptor model
5_validation_and_applicability_domain/  Y-randomisation, outlier rule, AD, external validation,
                                        similarity, deep-learning benchmark
6_interpretation/                       SHAP values, descriptor interactions, chemical-space errors
7_isomer_analysis/                      Isomer ranking, 839 isomer pairs, group-wise CV
8_extra/                                Organosilicon check, timing, package versions
data/dataset_public_5071.csv            Name, CAS, SMILES, formula, MW, source (NIST/DIPPR), split,
                                        Tb in K (blank for DIPPR), the 10 final descriptors
model/                                  final_model_lightgbm.txt, final_model_scaler.json,
                                        verify_nist_test.py, test_metrics_by_source.csv
optuna/                                 Optuna study databases (.db) and CSV exports of all trials
results/                                Per-compound and summary outputs
requirements_pinned.txt                 Exact package versions
```

The scripts were exported from the Google Colab notebooks used for the paper. Run them in the numbered order within each folder; each script states at the top what it reads and writes. File paths start with `./BP_FINAL_MODEL/...`; change them to your own folder layout.

## DIPPR data

Boiling points from the DIPPR 801 database are licence-restricted and are **not redistributed**. In every file, rows from DIPPR-sourced compounds have the experimental boiling point, and any column that would reveal it (residual, absolute error, experimental ΔTb), left **blank**. Compound identities, descriptors and model predictions are included. DIPPR values can be obtained with a DIPPR subscription; the complete dataset is available to editors and reviewers on request.

`model/verify_nist_test.py` reproduces the published metrics for the 855 NIST-sourced test compounds (R² = 0.9602, MAE = 12.07 K, RMSE = 17.67 K).

## Scripts and the results they produce

| Script | Output |
|---|---|
| `1_data_curation/01_raw_data_selection.py` | Stratified selection of DIPPR and NIST compounds (10 x 10 boiling point x molecular weight bins) |
| `1_data_curation/02_data_curation_pipeline.py` | Seven-step curation pipeline (Table S2) |
| `1_data_curation/03_distribution_analysis.py` | Boiling point and molecular weight distributions; descriptive statistics (Table 1) |
| `1_data_curation/04_functional_group_classification.py` | Functional-group classification (SMARTS) |
| `1_data_curation/05_functional_group_counts.py` | Functional-group definitions and counts (Table S3) |
| `1_data_curation/06_histograms_by_family.py` | Molecular mass and Tb histograms coloured by chemical family (Figure 2, Figure S1) |
| `2_descriptor_calculation/01_generate_descriptors.py` | RDKit 2D/3D and Mordred descriptors (2,054 calculated) |
| `3_descriptor_selection/01_variance_spearman_vif_filter.py` | Constant/non-numeric/zero, variance, Spearman and VIF filters (1,916 -> 94) |
| `3_descriptor_selection/02_random_forest_ranking.py` | Random Forest impurity ranking (94 -> top 35) |
| `3_descriptor_selection/03_bp_stratified_train_test_split.py` | Boiling-point-stratified 80/20 train/test split (10 bins, random_state = 42) |
| `3_descriptor_selection/04_apply_split_to_descriptor_sets.py` | Applies the same train/test split to the 94- and 35-descriptor sets |
| `3_descriptor_selection/05_check_split_files.py` | Checks the layout of the train/test files |
| `3_descriptor_selection/06_split_strata_table.py` | Boiling-point strata and train/test counts per bin |
| `3_descriptor_selection/07_backward_elimination_35desc.py` | Backward elimination from 35 descriptors for all five models |
| `3_descriptor_selection/08_optimal_n_summary_figure.py` | Summary figure of the optimal descriptor count per model |
| `3_descriptor_selection/09_lightgbm_top_descriptors.py` | Top descriptors of the LightGBM elimination checkpoint |
| `3_descriptor_selection/10_build_optimal_descriptor_files.py` | Builds the final 10-descriptor train/test files |
| `3_descriptor_selection/11_descriptor_set_significance.py` | Significance of the change in R2 with descriptor-set size (Table 6) |
| `3_descriptor_selection/12_elimination_trajectories_figure.py` | Backward-elimination trajectories of all five models (Figure 6) |
| `3_descriptor_selection/13_descriptor_mapping_2054.py` | Stage at which each of the 2,054 descriptors was removed or kept (Table S5) |
| `3_descriptor_selection/14_selection_robustness.py` | Robustness to the selection metric and tree algorithm (Table S13) |
| `3_descriptor_selection/15_training_only_selection_check.py` | Descriptor selection repeated on the training set only |
| `4_model_development/01_pipeline_35desc.py` | Optuna tuning and five models on the 35-descriptor set (Table 5, Table S6) |
| `4_model_development/02_pipeline_94desc.py` | Optuna tuning and five models on the 94-descriptor set (Table 5, Table S6) |
| `4_model_development/03_applicability_domain_from_checkpoint.py` | Applicability-domain analysis from a saved checkpoint |
| `4_model_development/04_compare_94_vs_35.py` | Comparison of the 94- and 35-descriptor sets (Figure 5) |
| `4_model_development/05_final_pipeline_10desc.py` | FINAL MODEL: tuning, five models, log transformation, applicability domain (Tables 3, 5, 7, 11) |
| `4_model_development/06_figure_performance_bars.py` | Performance bar chart |
| `4_model_development/07_figure_log_transform.py` | Original vs log-transformed target, 10 descriptors (Figure 7) |
| `4_model_development/08_heteroscedasticity.py` | Breusch-Pagan tests and residual-vs-fitted plots (Table S8, Figure S13) |
| `4_model_development/09_residual_statistics.py` | Residual statistics and predicted-vs-experimental figure (Table S12, Figure 4) |
| `4_model_development/10_ann_training_curves.py` | ANN training curves over five seeds (Figure S12) |
| `5_validation_and_applicability_domain/01_y_randomisation_100.py` | Y-randomisation, first 100 permutations |
| `5_validation_and_applicability_domain/02_y_randomisation_1000.py` | Y-randomisation extended to 1,000 permutations (Table S7, Figure S6) |
| `5_validation_and_applicability_domain/03_outlier_rule_sensitivity.py` | Sensitivity to the boiling-point outlier rule (Tables S17, S18) |
| `5_validation_and_applicability_domain/04_compounds_outside_ad.py` | Test compounds outside the applicability domain (Table S21) |
| `5_validation_and_applicability_domain/05_source_flags_and_model_export.py` | NIST/DIPPR source flags, public dataset and final model files |
| `5_validation_and_applicability_domain/06_external_validation_opera.py` | External validation on OPERA boiling-point data (Table 9) |
| `5_validation_and_applicability_domain/07_structural_similarity.py` | Structural similarity and error vs similarity (Table 10, Figure 9) |
| `5_validation_and_applicability_domain/08_deep_learning_benchmark.py` | Chemprop D-MPNN and CheMeleon benchmark (Table S20) |
| `6_interpretation/01_shap_values_export.py` | Per-compound SHAP values, training and test sets (Table S9) |
| `6_interpretation/02_descriptor_interactions.py` | SHAP interactions, Friedman H, additive model (Table S19, Figure S11) |
| `6_interpretation/03_chemical_space_errors.py` | PCA/t-SNE chemical-space map of errors (Table S14, Figure S10) |
| `7_isomer_analysis/01_isomer_analysis.py` | Isomer ranking analysis (Table 12, Figures S7-S9) |
| `7_isomer_analysis/02_export_isomer_pairs.py` | Export of the 839 isomer pairs |
| `7_isomer_analysis/03_isomer_groupwise_cv.py` | Isomer groups kept together in cross-validation (Table 13) |
| `8_extra/01_organosilicon_check.py` | Predictions for organosilicon compounds (Table S11) |
| `8_extra/02_prediction_timing.py` | Wall-clock time of a single prediction |
| `8_extra/03_software_versions.py` | Package versions |

## Reproducibility notes

- The scripts were run in Google Colab (Python 3.12/3.13); package versions are in `requirements_pinned.txt`.
- Re-running the variance/Spearman/VIF and Random Forest ranking steps from scratch reproduces 88 of the 94 VIF-filtered and 21 of the 35 top-ranked descriptors. The saved sets used in the paper are provided in `results/` (`stage2_reduced.csv`, `top35_descriptors.csv`, `rf_descriptor_ranking.csv`, `descriptor_mapping_2054.xlsx`).
- All ten final descriptors are two-dimensional. Calculating them after 3D embedding (as in `2_descriptor_calculation/01_generate_descriptors.py`) or directly on the hydrogen-added 2D molecule (as in the web application) gives identical values.

## Main result files (`results/`)

SHAP values (training and test), isomer pairs and group-wise predictions, 1,000 Y-randomisation R² values, ANN training histories, descriptor-selection trace and mapping, cross-evaluation of descriptor sets, outlier-rule sensitivity, heteroscedasticity tests, descriptor interactions, chemical-space coordinates and regional errors, external-validation predictions, similarity analysis, deep-learning benchmark, organosilicon check and compounds outside the applicability domain.

External validation data: OPERA (Mansouri, K. et al. *J. Cheminform.* **2018**, 10, 10).

## Licence

Code: MIT (see `LICENSE`).

## Contact

Muhammad Iqbal Javed, School of Applied Sciences, University of Huddersfield, Huddersfield HD1 3DH, UK – muhammadiqbal.javed@hud.ac.uk
