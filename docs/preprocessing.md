# Preprocessing

`chart preprocessing` turns the per-well single-cell tables into the screen-wide cell table that the archetype analysis starts from. Within each well it removes cells that fail quality control; across wells it normalises the features, removes outlier cells within each perturbation, and removes guides that do not behave like the rest of their perturbation or, for controls, like the other controls. Its inputs are described under [Your data](data.md).

## Usage

```
chart preprocessing --config my_screen.yaml --all-wells
chart preprocessing --config my_screen.yaml --well A1 --steps qc filter
```

**Required**

| Argument | Value | Notes |
|---|---|---|
| `--config` | a path | The configuration file, in YAML or JSON |
| `--well` or `--all-wells` | for `--well`, one or more wells, separated by spaces | The wells to process, named as described under [Your data](data.md); `--all-wells` processes every well of the screen |

**Optional**

| Argument | Value | Default | Notes |
|---|---|---|---|
| `--steps` | one or more steps from the table below, or `all` | `all` | The steps to run |
| `--save-plots` | | off | Write the diagnostic plots; without it, none are written |
| `--plots-dir` | a path | each step's own, under `local_output_dir` | Write every plot to this directory instead |
| `--continue-on-error` | | off | Log a failed well or step and carry on with the rest |
| `--n-jobs` | `-1`, or a number of cores | the top-level `n_jobs`, else `-1` | The cores `normalize` and `guides` use; `-1` uses every core available to the run |
| `--quiet` | | off | Log warnings and errors only; this also hides the note that `--well` held back the across-wells steps |
| `--components` | one or more of `plots`, `segmentation`, `pheno`, `pheno_sbs`, `filters`, or `all` | `all` | The quality-control components `qc` runs; `filters` writes the lists. A component whose prerequisites, such as image directories or SCALLOPS, are missing is skipped with a warning, or stops the run when named |
| `--targets` | one or more of `objects`, `features`, or `all` | `all` | The tables `filter` applies the filter lists to |
| `--level` | one of `cell`, `guide`, `gene` | `cell` | The identifiers kept in the index of the tables `combine` writes; the rows stay one per cell. Only `cell` keeps each cell's label, which `chart archetypes` needs |
| `--validate` | | off | Plot five features chosen at random in each well, before and after `normalize`; written only with `--save-plots` |
| `--aggregate` | one or more of `guide`, `gene` | none for `normalize`, `guide` for `outliers` | Tables summarised per guide or per perturbation that `normalize` and `outliers` also write. This replaces both steps' defaults; `guides` reads the per-guide table `outliers` writes, so include `guide` |
| `--skip-prefilters` | one or more of `missing`, `variation`, `incomplete` | none | Prefilters `outliers` does not run (see Method), when they leave no cells or no features |
| `--aggregation-method` | one of `median`, `mean`, `std` | `median` | The statistic the per-guide and per-perturbation tables are summarised with. `guides` scores guides on their summaries, so this also changes which guides are removed |

## Steps

| Step | Runs | Output |
|---|---|---|
| `qc` | per well | Lists of cells failing quality control |
| `filter` | per well | The well's tables without those cells, reduced to the feature columns |
| `combine` | across wells | One objects table and one feature table for the screen |
| `normalize` | across wells | Features quantile-normalised across wells |
| `outliers` | across wells | Outlier cells removed within each perturbation |
| `guides` | across wells | Unreliable guides removed; cell, guide and perturbation tables |

`--steps` takes one or more step names from the table, separated by spaces; without it, every step runs. Only the named steps run, always in the table's order, whatever order they are given in. Each step reads what the step before it wrote: from this run if that step ran, otherwise from an earlier run. A step whose input is missing stops with an error. After changing a setting, rerun the step that reads it and every step after it, for instance, after changing `guide_filtering.cosine_similarity_threshold`:

```
chart preprocessing --config my_screen.yaml --all-wells --steps guides
```

`--all-wells` without `--steps` runs every step in one command. With `--well`, only the per-well steps run unless the across-wells steps are named in `--steps`, since those write one table for the whole screen. The per-well steps can therefore run as separate jobs, one per well, followed by a single run across wells:

```
chart preprocessing --config my_screen.yaml --well A1 --steps qc filter --save-plots
chart preprocessing --config my_screen.yaml --all-wells --steps combine normalize outliers guides --save-plots
```

If the step before it runs in the same command but fails or writes nothing, a step does not run rather than read an older table.

The log is written to `preprocessing/bywell/qc_reports/` under `local_output_dir`.

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

The `pheno` and `pheno_sbs` components read their images through SCALLOPS, from these keys under `preprocessing_bywell.global` or `preprocessing_bywell.wells.<well>`:

| Key | Contents |
|---|---|
| `pheno_dir` | Phenotyping images |
| `dapi_channel_names` | The DAPI channel of each phenotyping round; `pheno` needs at least two |
| `pheno_sbs_registered_dir` | Phenotyping images registered to the sequencing images |
| `pheno_sbs_dapi_channel` | The DAPI channel among those, by name or position; by default the last of `dapi_channel_names` |
| `sbs_dir` | Sequencing images; DAPI is read from the first channel of the first cycle |
| `sbs_feature_dir` | Holds `nuclei/<well>-objects.parquet`, the nuclei in sequencing coordinates |

Filters computed elsewhere are added under `preprocessing_bywell.precomputed_filters`, one entry per filter, keyed by its name. Each entry gives `path`, a directory; `file_pattern`, default `{well}-objects.parquet`; optionally `threshold_column` and `threshold`, keeping the rows at or above the threshold; and `role`, `inclusion` (default) or `exclusion`. The table's index has to hold the cell labels.

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
| `'...' was requested for well ... but ...` | A component named in `--components` lacks its images, settings or SCALLOPS |
