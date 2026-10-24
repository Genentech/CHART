# Configuration

Every stage reads one YAML file, given with `--config`. [`example_config.yaml`](https://github.com/Genentech/CHART/blob/main/example_config.yaml), reproduced at the end of this page, is the starting point: copy it, set the two output directories, and adjust what the screen requires. Each key is commented there, and the values shown are the defaults, except the channel lists, which are examples.

## How the file is read

| Section | Read by |
|---|---|
| top level: `data_output_dir`, `local_output_dir`, `n_jobs` | every stage |
| `schema` | every stage |
| `preprocessing_bywell` | `chart preprocessing`; the image locations also by the `crops` step |
| `preprocessing_allwells`, `guide_filtering` | `chart preprocessing` |
| `archetypes` | `chart archetypes`, and the channels and removed archetypes by every later stage |
| `annotation`, `effects`, `programs` | the stage of the same name |

Only the two output directories are required. Tables are written under `data_output_dir`, and reports, plots and filter lists under `local_output_dir`. A relative path in the file is taken under the corresponding output directory, and image directories under `scallops_dir` where it is set; absolute paths and URLs such as `s3://` are used as given. A key CHART does not recognise is reported with a warning, `Unknown config key '...' in ...; it will be ignored`, so a misspelt key does not pass silently. Command-line options such as `--n-jobs` and `--parallel-channels` override the corresponding keys for one run.

## Settings by importance

**Required.** `data_output_dir` and `local_output_dir`.

**Describing the screen.** These depend on the data, and their defaults fit only the reference dataset. They are described under [Your data](data.md), and the channels under [Archetypes](archetypes.md).

- `schema`: the identifier columns, `feature_patterns`, the controls `control_gene` and `control_prefix`, and the coordinate, bounding-box and area columns that quality control and the crops read
- `preprocessing_bywell`: the input location and file names, `merged_dir`, `objects_pattern`, `features_pattern`, or `premerged` and `merged_pattern`; and the image locations under `global` and `wells`
- `archetypes`: `channels`, `all_channels` and `image_channels`

**Decided during the analysis.** These are set after inspecting the outputs of an earlier step.

- `archetypes.k`, and `archetypes.npc`, after checking the recovery curve and the variance elbow ([Archetypes](archetypes.md))
- `archetypes.archetypes_to_remove`, after inspecting the crops ([Archetypes](archetypes.md))
- `annotation.normal_archetypes`, optionally, after inspecting the crops; otherwise the most abundant archetype is used ([Annotation](annotation.md))

**Analysis parameters.** The defaults are those of the published analysis. Changing them changes the results, so they are best changed deliberately and recorded.

- `preprocessing_bywell.thresholds` and `guide_filtering`: the quality-control and guide-filtering thresholds ([Preprocessing](preprocessing.md))
- `archetypes.params`: the metacell graining, `gamma` and `k.knn` ([Archetypes](archetypes.md))
- `effects.n_perm` and `effects.trimmed_alpha` ([Effects](effects.md))
- `programs.k`, `programs.n_modules` and `programs.qval_threshold` ([Programs](programs.md))
- `annotation.model` ([Annotation](annotation.md))

**Resources.** `n_jobs`, `effects.n_jobs` and `archetypes.parallel_channels` set how many cores and channels a run uses, and do not change the results ([Running on a cluster](cluster.md)).

**Rarely changed.** The remaining keys have defaults that suit most screens: the output subdirectories (`outputs_dir`, `reports_dir`), `drop_unassigned` and `exclude_patterns`, the normalisation and outlier settings in `preprocessing_allwells`, the archetype sample sizes and `selected_params`, the crop display (`top_cells`, `display_percentiles`, `display_ranges`), the seeds, and the annotation's `similarity_threshold` and `max_pairs_per_archetype`.

## The example file

```{literalinclude} ../example_config.yaml
:language: yaml
```
