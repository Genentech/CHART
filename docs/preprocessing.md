# Preprocessing

`chart preprocessing` turns the per-well single-cell tables into the screen-wide cell table that the archetype analysis starts from. Within each well it removes cells that fail quality control; across wells it normalises the features, removes outlier cells within each perturbation, and removes guides that do not behave like the rest of their perturbation or, for controls, like the other controls. Its inputs are described under [Your data](data.md).

## Usage

```
chart preprocessing --config my_screen.yaml --all-wells
chart preprocessing --config my_screen.yaml --well A1 --steps qc filter
```

| Step | Runs | Output |
|---|---|---|
| `qc` | per well | Lists of cells failing quality control |
| `filter` | per well | The well's tables without those cells, reduced to the feature columns |
| `combine` | across wells | One objects table and one feature table for the screen |
| `normalize` | across wells | Features quantile-normalised across wells |
| `outliers` | across wells | Outlier cells removed within each perturbation |
| `guides` | across wells | Unreliable guides removed; cell, guide and perturbation tables |

The steps run in order. With `--well`, only the per-well steps run unless the across-wells steps are named in `--steps`, since those write one table for the whole screen. The per-well steps can therefore run as separate jobs, one per well, followed by a single `--all-wells --steps combine normalize outliers guides`. When steps run together, a step whose input was not written earlier in the same run stops rather than read an older table; naming it alone in `--steps` uses the existing table.

`--components` selects the quality-control components: `plots`, `segmentation`, `pheno`, `pheno_sbs`, and `filters`, which writes the lists. A component whose prerequisites are missing, such as image directories or scallops, is skipped with a warning, or stops the run when named in `--components`.

Plots are written only with `--save-plots`; `--validate` adds before-and-after distributions of the normalisation. `--continue-on-error` carries on with the remaining wells after a failure. `normalize` and `guides` use every core available to the run; the top-level `n_jobs`, or `--n-jobs`, sets a number instead. The log is written to `preprocessing/bywell/qc_reports/` under `local_output_dir`.

## Outputs

Tables are written under `data_output_dir`; filter lists, records and plots under `local_output_dir`.

| File | Under | Contents |
|---|---|---|
| `preprocessing/bywell/filters/<well>_*.parquet` | local | The well's filter lists, and `<well>_filters.json`, the components run and the lists written |
| `preprocessing/bywell/filtered/<well>-objects.parquet`, `<well>-features.parquet`, `<well>.parquet` | data | The well's filtered objects, features, and both joined |
| `preprocessing/allwells/combined/allwells-objects.parquet`, `allwells-features.parquet` | data | All wells stacked |
| `preprocessing/allwells/normalized/allwells-features_cell.parquet` | data | Normalised features |
| `preprocessing/allwells/outlier_filtered/allwells-features_cell_outlier_filtered.parquet` | data | Outlier cells removed; `_guide_outlier_filtered` holds the per-guide medians |
| `preprocessing/allwells/filters/outlier_filters.parquet` | local | Every cell removed and why; `outlier_filters.json` records the settings |
| `guide_filtering/guide_filtered/allwells-features_cell.parquet` | data | The final cell table, read by `chart archetypes` |
| `guide_filtering/guide_filtered/allwells-features_guide.parquet`, `allwells-features_gene.parquet` | data | Per-guide medians, and per-perturbation medians of those |
| `guide_filtering/filters/guide_filters.parquet` | local | Every guide removed and why; `guide_filters.json` records the settings |

Cells are indexed by `Label`, `Guide`, `Gene` and `Well` from `combine` on.

## Configuration

```yaml
preprocessing_bywell:
  thresholds:
    segmentation_error_threshold: 15   # cytoplasm-to-nucleus area ratio
    pheno_correlation: 0.8             # DAPI correlation across phenotyping rounds
    pheno_sbs_correlation: 0.8         # DAPI correlation, phenotyping against sequencing

preprocessing_allwells:
  normalization_type: quantile
  contamination: auto                  # or the proportion of cells to flag
  random_state: 42
  n_estimators: 100
  missing_value_threshold: 100000      # a count of cells, not a fraction
  rcv_threshold: 0.01

guide_filtering:
  cosine_similarity_threshold: 0.2
  min_cells_per_guide: 50
```

The values shown are the defaults. The stage also reads `schema`, and the input settings described under [Your data](data.md).

The `pheno` and `pheno_sbs` components read their images through scallops, from these keys under `preprocessing_bywell.global` or `preprocessing_bywell.wells.<well>`:

| Key | Contents |
|---|---|
| `pheno_dir` | Phenotyping images |
| `dapi_channel_names` | The DAPI channel of each phenotyping round; `pheno` needs at least two |
| `pheno_sbs_registered_dir` | Phenotyping images registered to the sequencing images |
| `pheno_sbs_dapi_channel` | The DAPI channel among those, by name or position; by default the last of `dapi_channel_names` |
| `sbs_dir` | Sequencing images; DAPI is read from the first channel of the first cycle |
| `sbs_feature_dir` | Holds `nuclei/<well>-objects.parquet`, the nuclei in sequencing coordinates |

Filters computed elsewhere are added under `preprocessing_bywell.precomputed_filters`, one entry per filter, keyed by its name. Each entry gives `path`, a directory; `file_pattern`, default `{well}-objects.parquet`; optionally `threshold_column` and `threshold`, keeping the rows at or above the threshold; and `role`, `inclusion` (default) or `exclusion`. The table's index has to hold the cell labels.

## Method

**Quality control.** `segmentation` excludes cells whose cytoplasmic area exceeds `segmentation_error_threshold` times their nuclear area. `pheno` correlates, within each nucleus's bounding box, the first phenotyping round's DAPI image with each later round's, and keeps the cells whose mean Pearson correlation reaches `pheno_correlation`; a poor correlation marks a cell misregistered between rounds. `pheno_sbs` does the same between the phenotyping DAPI image registered to the sequencing images and the sequencing DAPI image, which ties a cell's phenotype to its guide assignment. A cell is kept if it appears in every inclusion list and in no exclusion list.

**Normalisation.** Each feature is quantile-normalised across wells: a well's values are ranked, ties taking their average rank, and mapped by linear interpolation onto the feature's distribution pooled over the screen. Missing values take no part and stay missing.

**Outliers.** Three prefilters run first: features with more than `missing_value_threshold` missing values are dropped, then features whose robust coefficient of variation (normal-scaled median absolute deviation over the median) is below `rcv_threshold`, then cells with any missing value. An isolation forest (`n_estimators` trees, seeded by `random_state`) is then fitted to the cells of each perturbation separately, and the cells it flags are removed; perturbations with fewer than two cells are not tested. `contamination: auto` uses the forest's own score threshold.

**Guides.** The step works on per-guide medians of the outlier-filtered cells. Control guides are flagged when at least two of three detectors agree: an isolation forest on the features, and a local outlier factor (20 neighbours) and an elliptic envelope in a PCA space. The space is fitted to all guides after standardisation and keeps the components up to 90% of the variance. A second PCA space is fitted to the remaining guides and centred on the control guides, and each perturbation guide is scored by its mean cosine similarity to the other guides of its perturbation; guides below `cosine_similarity_threshold` are removed. Perturbations with a single guide cannot be scored and are kept. Last, guides with fewer than `min_cells_per_guide` cells left are removed. A guide's profile is the median of its cells, and a perturbation's the median of its guides', so that a guide with many cells does not dominate.

## Troubleshooting

| Message | Cause and remedy |
|---|---|
| `well ... has no QC filter lists in ...; run the qc step first` | QC wrote no list for the well; without images only `segmentation` produces one, and only if it flags a cell. Without `--well` the well is skipped with a warning instead |
| `Every one of the ... objects was filtered out ...` | The filter lists and the tables identify cells differently, often by label type, or the lists belong to another well |
| `No feature columns: none of the ... columns match any of the patterns ...` | `schema.feature_patterns` matches no column |
| `... of the ... columns to normalise hold something other than numbers ...` | `schema.feature_patterns` selects a text column; narrow the patterns |
| `Not normalising: ...`, `Not filtering outliers: ...`, `Not filtering guides: ...` | The previous step wrote nothing in this run; fix it, or name the step alone to use an existing table |
| `Prefiltering left ... cells and ... columns ...` | The prefilters removed everything; check `missing_value_threshold` and `rcv_threshold`, or `--skip-prefilters` |
| `No control guides among ... rows ...` | No perturbation matches `schema.control_gene` or `schema.control_prefix` |
| `Guide filtering removed every cell ...` | `cosine_similarity_threshold` or `min_cells_per_guide` is too strict for the screen |
| `'...' was requested for well ... but ...` | A component named in `--components` lacks its images, settings or scallops |
