# Boiling Point Prediction Model (QSPR)

A machine-learning QSPR model for predicting normal boiling point (K) of
organic compounds from molecular descriptors. Best model: **LightGBM**
(Test R² = 0.9596, MAE = 12.32 K), trained on 10 optimal descriptors,
with Applicability Domain and Y-randomization validation.

> Manuscript prepared for submission to *J. Chem. Inf. Model.* — see the
> Citation section below. The paper DOI will be added once assigned by the
> journal.

## Repository structure

The pipeline is organized to match the stages described in the paper's
Methods section:

```
1_data_preparation/                          Dataset selection & curation, EDA
2_descriptor_calculation/                    RDKit2D + Mordred descriptor generation
3_feature_selection/                         Variance/Spearman/VIF filtering, Boruta,
                                              RF ranking, backward elimination
4_parameter_optimization_and_model_development/
                                              Optuna-tuned Linear/RF/XGBoost/LightGBM/ANN,
                                              final model selection, SHAP
5_applicability_domain_and_y_randomization/  Williams-plot AD analysis, Y-scrambling test
extra/                                       Software version capture, isomer-level validation
data/                                        Compound identity, descriptors, and train/test
                                              splits (DIPPR boiling point values redacted --
                                              see data/README.md and Data Availability below)
```

Each script corresponds to one step of the pipeline and is numbered in the
order it should be run. `4_.../final_model_pipeline.py` is the authoritative,
final-run version of the modeling step — it performs hyperparameter
optimization, model training/comparison, and includes the Applicability
Domain analysis inline for the optimal descriptor set. Y-randomization is
deliberately kept separate (it's a slow, many-iteration validation step
unrelated to model selection) — run it via
`5_.../run_y_randomization.py`, which loads everything it needs straight
from the checkpoint saved by the modeling script, no retraining required.
The AD analysis also has a standalone counterpart
(`5_.../run_applicability_domain.py`) for re-running just that analysis
from a saved checkpoint.

## Installation

```bash
git clone https://github.com/iqbaljaved54568/BP-boiling-point-model.git
cd BP-boiling-point-model
pip install -r requirements.txt
```

This project was originally developed in Google Colab. `extra/software_versions.py`
captures the exact package versions used for the results reported in the paper
(see the SI, Table S8 for the printed version list).

## How to reproduce

Run the numbered scripts in each folder in order:

1. `1_data_preparation/` — builds the curated dataset from raw source data
2. `2_descriptor_calculation/` — computes molecular descriptors
3. `3_feature_selection/` — reduces to the final 10-descriptor optimal set
4. `4_parameter_optimization_and_model_development/final_model_pipeline.py` — trains,
   tunes, and selects the final model
5. `5_applicability_domain_and_y_randomization/` — validates the model (AD, Y-scrambling)

Each script currently references its input/output paths under `data/` —
update these to match where you place your own files.

## Data availability

Compound identity (SMILES, CAS number, name, formula), all computed
molecular descriptors, and NIST-sourced boiling point values are included
in `data/` (see `data/README.md` for the full breakdown).

The raw training data draws in part from DIPPR (Design Institute for
Physical Properties), which is subscription/license-restricted. **DIPPR's
actual measured boiling point values have been redacted** from every file
in `data/` where they would otherwise appear (flagged via a `data_source`
column) -- everything else about those compounds (identity, descriptors) is
included. To obtain the withheld DIPPR boiling point values, you'll need
your own DIPPR subscription/access, or the curated dataset is available
from the corresponding author (Muhammad Iqbal Javed,
muhammadiqbal.javed@hud.ac.uk) upon reasonable request.

## Citation

If you use this code, please cite:

```
Javed, M. I.; Waters, L.; Belton, D.; Cooke, D. Interpretable and Parsimonious
QSPR Modelling of the Normal Boiling Point of Structurally Diverse Organic
Compounds. J. Chem. Inf. Model. 2026. DOI: [paper DOI — to be added upon publication]

Code archive: DOI: [Zenodo DOI — to be added after archiving, see publishing guide]
```

**Corresponding author**: Muhammad Iqbal Javed, School of Applied Sciences,
University of Huddersfield, Huddersfield, HD1 3DH, United Kingdom.
Email: muhammadiqbal.javed@hud.ac.uk

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
