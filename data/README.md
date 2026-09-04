# Data

This folder contains the compound identity, curation, descriptor, and
train/test split data used in the paper -- with one deliberate exception,
explained below.

## What's included

```
raw/
  NIST_DIPPR_raw_6000.csv       6,000-compound combined raw dataset (Step 0
                                 of data curation), NIST- and DIPPR-sourced,
                                 ~274 overlapping/duplicate entries between
                                 the two included as-is (see curation script)
  DIPPR_metadata_1500_no_BP.csv 1,500-compound DIPPR-sourced stratified
                                 subset -- identity only (Name, CAS, SMILES,
                                 Formula), no property values
  manual_exclusion_list.csv     73 compounds manually excluded during
                                 curation (Step 1), with documented reasons
                                 implicit in compound identity (e.g.
                                 elements, unstable/reactive species)

descriptors/
  top35_descriptors.csv         5,071-compound dataset with the top-35
                                 descriptor set (post first-round variance /
                                 Spearman / VIF / Boruta / RF-ranking
                                 selection, pre backward elimination)

train_test_split/
  train_optimal.csv             4,056-compound training split, final
                                 10-descriptor optimal set
  test_optimal.csv              1,015-compound test split, final
                                 10-descriptor optimal set
  train_35desc.csv               4,056-compound training split, 35-descriptor
                                 set
  test_35desc.csv                1,015-compound test split, 35-descriptor set
```

## Important: DIPPR boiling point values are redacted

DIPPR (Design Institute for Physical Properties) data is
subscription/license-restricted. Its structural/compound identity (SMILES,
CAS number, name, formula) is included, and so are all computed molecular
descriptors (these are derived independently from the public SMILES string
by anyone running `2_descriptor_calculation/`, not DIPPR's proprietary
contribution) -- but **the actual measured boiling point value for any
compound sourced from DIPPR has been removed** from every file above where
it would otherwise appear.

Each affected file has a `data_source` column (`DIPPR` or `NIST`) so it's
transparent exactly which rows are affected. Rows with `data_source ==
DIPPR` have `NaN`/blank in the `boiling point (K)` column.

| File | Total rows | DIPPR-origin rows (BP redacted) |
|---|---:|---:|
| `raw/NIST_DIPPR_raw_6000.csv` | 6,000 | 1,775 |
| `raw/manual_exclusion_list.csv` | 73 | 36 |
| `descriptors/top35_descriptors.csv` | 5,071 | 1,226 |
| `train_test_split/train_optimal.csv` | 4,056 | 1,003 |
| `train_test_split/test_optimal.csv` | 1,015 | 223 |
| `train_test_split/train_35desc.csv` | 4,056 | 1,003 |
| `train_test_split/test_35desc.csv` | 1,015 | 223 |

To obtain the withheld DIPPR boiling point values (e.g. to fully reproduce
model training), you'll need your own DIPPR subscription/access, or contact
the corresponding author -- see the main README's Data Availability
section.

## Known data-quality notes

- **One duplicate compound across the 35-descriptor train/test split**:
  CAS `13027-88-8` appears in both `train_35desc.csv` (as
  "alpha-HYDROXYISOBUTYRAMIDE", SMILES `CC(C)(O)C(N)=O`) and
  `test_35desc.csv` (as "2-hydroxy-2-methylpropanamide", SMILES
  `CC(C)(O)C(=N)O`) -- the same real compound entered under two different
  tautomer SMILES and name spellings. This was not deduplicated here; with
  1 compound out of 5,071 the effect on reported metrics is expected to be
  negligible, but it's flagged here for transparency and worth a look
  before any future retrain.
- Several rows across files have no recorded CAS number (69 in
  `train_35desc.csv`, 17 in `test_35desc.csv`) -- these are not the
  duplicate-CAS issue above, just genuinely missing identifiers in the
  source data.

## Known gaps in this folder

- An outliers/compounds-flagged-outside-AD list was mentioned as a
  candidate addition but not yet provided.

If you have it, it can be added the same way -- see the main README for
how this repository is organized.
