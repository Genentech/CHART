# Archetypes

`chart archetypes` describes each channel's single-cell morphology as a mixture of archetypes, extreme morphological states between which every cell lies. It builds a PCA space from the channel's features, groups the cells into metacells, fits archetypal analysis to the metacells, and weights every cell on each archetype. The weights are the input to `chart effects` and `chart programs`; the last step crops each archetype's most heavily weighted cells from the images, for inspection and for `chart annotation`.

```{image} ../images/archetypes_diagram.svg
:alt: How chart archetypes finds archetypes and weights cells on them
:width: 100%
:align: center
```

## Usage

```
chart archetypes --config my_screen.yaml --all-channels
chart archetypes --config my_screen.yaml --channel TOM20 --steps selectk weights crops
```

| Step | Runs | Output |
|---|---|---|
| `pca` | Python | The channel's PCA space, centred on the controls |
| `npc` | R | The number of principal components kept |
| `metacells` | R, per parameter pair | Metacells, and their coordinates |
| `evaluate` | R | A comparison of the metacell partitions; optional, as nothing downstream reads it |
| `archetypes` | R, per parameter pair | Repeated archetype fits for each number of archetypes from 6 to 20 |
| `selectk` | R | The number of archetypes, K |
| `weights` | R | Every cell's weight on each of the K archetypes |
| `crops` | Python | Each archetype's top cells, cropped from the images |

`--steps` takes one or more step names from the table, separated by spaces; without it, every step runs. Only the named steps run, always in the table's order, whatever order they are given in. Each step reads what the step before it wrote: from this run if that step ran, otherwise from an earlier run. A step whose input is missing stops with an error. After changing a setting, rerun the step that reads it and every step after it, for instance, after setting `archetypes.k`:

```
chart archetypes --config my_screen.yaml --channel TOM20 --steps selectk weights crops
```

The R steps need the environment described under [Installation](installation.md); the stage checks that R can load every package before the first step. `--dry-run` prints every command the stage would run, and needs no R. `--continue-on-error` carries on with the remaining steps and channels after a failure.

Channels run one at a time. `archetypes.parallel_channels`, or `--parallel-channels`, runs several at once, each in its own process and logging to `archetypes/<channel>/` under `local_output_dir`. Memory, not cores, limits how many fit: the `metacells` step needed about 62 GB for one channel of 875,000 cells. Channels can also run as separate jobs with `--channel`.

> [!IMPORTANT]
> `selectk` chooses K automatically and says so in the log. Check its choice against `evalK_recovery_plot.png` before going further; to overrule it, set `archetypes.k` for the channel and rerun `--steps selectk weights crops`.

`crops` needs SCALLOPS, the image locations described under [Your data](data.md), and the channel's position in the images under `archetypes.image_channels`. Without them the step is skipped with a warning, or stops the run under `--steps crops`. Once the crops are drawn, curate the archetypes, as described under Curating archetypes below, before running `chart effects`.

## Configuration

```yaml
archetypes:
  channels: [TOM20, Golgin97]
  all_channels: [DAPI1, TOM20, Golgin97, CALR, EEA1]
  outputs_dir: archetypes/      # under data_output_dir
  reports_dir: archetypes/      # under local_output_dir
  params: [[30, 10]]            # gamma and k.knn pairs
  # selected_params: [30, 10]   # the pair selectk and weights use; by default the first
  # npc: {TOM20: 14}            # by default the elbow of the variance curve
  # k: {TOM20: 12}              # by default the knee of the recovery curve
  coverage_sample_size: 12000
  eval_sample_size: 8000
  fit_sample_size: 12000
  image_channels: {TOM20: 8, Golgin97: 9}  # position among the image channels, from 0
  top_cells: 10
  # display_percentiles: {TOM20: [0, 99.9]}  # by default 0 and 99.5
  # display_ranges: {TOM20: {A1: [105, 4000]}}
  parallel_channels: 1
```

Where a key has a default, the value shown is the default and that of the published analysis. `channels` are the channels analysed; `all_channels`, every channel in the feature table, is required by `pca`: a channel's features are those that name no other channel, so a channel left out of the list keeps its features in every other channel's space. `display_percentiles` sets, per channel, the pixel percentiles between which the stain is shown, and `display_ranges` fixes the range for particular wells. The stage also reads `schema`, the guide-filtered cell table from [Preprocessing](preprocessing.md), and, for `crops`, the combined objects table and the image locations.

## Outputs

Each channel's data are written to `archetypes/<channel>/` under `data_output_dir`, and its reports to `archetypes/<channel>/` under `local_output_dir`. File names carry the parameters, `gamma`, `knn`, `npc` and `k`.

| File | Under | Contents |
|---|---|---|
| `cell_pca_centered.parquet` | data | The cells' principal components, with `Well`, `Label`, `Guide` and `Gene` |
| `npc.txt`, `k.txt` | data | The number of components and of archetypes in use |
| `supercells-<channel>_gamma…_knn…_npc….rds`, `.parquet` | data | Metacell membership, and metacell coordinates |
| `archetype_robust_runs_gamma…_knn…_npc….rds` | data | Every archetype fit, with its errors and stability measures |
| `robust_archweights-<channel>_k…_gamma…_knn…_npc….parquet` | data | One row per cell, a column per archetype, `Archetype1` to `ArchetypeK`, read by `chart effects` and `chart programs` |
| `npc_elbow.pdf` | local | The variance curve and the components kept |
| `space_evals.csv` | local | Size, dispersion and separation of each partition into metacells |
| `evalK_recovery_curves.csv`, `evalK_recovery_plot.png` | local | Reconstruction error against K, and why any K was excluded |
| `robust_archweights-….pdf` | local | The distribution of each archetype's weights over cells |
| `archetype_total_weights.png` | local | Each archetype's total weight over all cells, written by `crops` |
| `crops/archetype<N>_<channel>_grid_<row>.png` | local | Each archetype's top cells, five to an image |
| `crops/top_cells.parquet` | local | The cells shown, with their weights and bounding boxes |

## Curating archetypes

Some archetypes capture technical variation rather than morphology. These should be identified from the crops and excluded before testing. Each grid shows five of an archetype's most heavily weighted cells, `top_cells` in all: the channel's stain in inverted grey, the cell outlined in red and its nucleus in blue, and above each the weight, well and cell label. Archetypes to exclude are typically those whose top cells reflect segmentation errors, for instance outlines enclosing several cells or a fragment of one, or staining artefacts such as debris, precipitates or out-of-focus regions. `top_cells.parquet` gives each cell's well, which shows whether an archetype is drawn from a single well.

Excluded archetypes are listed under `archetypes.archetypes_to_remove`, grouped under category names of your choice; only the archetype names, `<channel>_ArchetypeN`, are read:

```yaml
archetypes:
  archetypes_to_remove:
    seg_err: [TOM20_Archetype1]
    staining_artifacts: [TOM20_Archetype4]
```

`chart effects` does not test them, and since all tests are corrected together, removing one also changes the q-values of the rest; `chart annotation` does not describe them. The numbering belongs to one fit: changing K, `npc` or the selected parameter pair renumbers the archetypes, and the list has to be made again.

The crops, with `archetype_total_weights.png`, are also where each channel's normal archetypes for `chart annotation` are chosen; see [Annotation](annotation.md).

## Method

**Channel space.** A channel's features are the columns of the guide-filtered cell table that name no other channel in `all_channels` and measure no shape. They are standardised, reduced to the principal components covering 90% of the variance, and centred on the centroid of the control cells.

**Components.** `npc` is the elbow of the log variance per component over the first 30 components, the point furthest from the chord joining the curve's ends, held between 10 and 30.

**Metacells.** SuperCell groups the cells on a k-nearest-neighbour graph of their first `npc` components, with `k.knn` neighbours and a graining level `gamma`, the ratio of cells to metacells. A metacell's coordinates are the mean of its cells'. With several pairs under `params`, `evaluate` tabulates for each partition the metacell sizes, the dispersion within metacells, and the distance between neighbouring metacells relative to their radius, as a basis for `selected_params`.

**Archetype fits.** On the standardised metacell coordinates, archetypal analysis is fitted eight times for each K from 6 to 20, each fit the best of 10 random starts. Each fit uses `fit_sample_size` metacells: 70% from a coverage set of `coverage_sample_size` metacells chosen by farthest-point sampling, the rest drawn with probability proportional to rarity, the distance to the 30th nearest metacell, raised to 1.5. Every fit is scored by the residual sum of squares of a shared evaluation sample of `eval_sample_size` metacells, half from the coverage set and half rarity-weighted, projected onto the simplex of its archetypes. For each K, the fit with the lowest error is the reference; drift is the 90th percentile of the distances between its archetypes and those of the other fits, matched by Hungarian assignment, and support is the smallest number of evaluation metacells weighted above 0.3 on any one archetype.

**Choice of K.** For each K, the evaluation metacells' weights on the reference archetypes reconstruct their coordinates in all components, with the archetypes' positions in that space fitted by least squares; the root-mean-square error, discarding the worst 10% of metacells, forms the recovery curve. K values with an unsupported archetype, or with drift above twice the median drift, are excluded, and K is the knee of the remaining curve with both axes scaled to [0, 1]. The error decreases with K, so it has no minimum to choose.

**Weights.** K archetypes are fitted once to all metacells, unstandardised, as the best of 20 random starts with a fixed seed, so a rerun on the same metacells gives the same archetypes. Each cell's weights are the convex coefficients, non-negative and summing to one, that best reconstruct its first `npc` components from the archetypes.

**Crops.** Each archetype's `top_cells` most heavily weighted cells are cropped at their bounding box with an 8-pixel margin. A well's display range lies between two percentiles of its pixels, sampled in five 5000-pixel patches, one central and one per quadrant, inside a 10% margin.

## Troubleshooting

| Message | Cause and remedy |
|---|---|
| `Rscript is not on PATH ...` or `R cannot load ...` | Activate the environment described under [Installation](installation.md) |
| `The pca step needs every channel the feature table holds ...` | Set `archetypes.all_channels` |
| `None of the ... columns belong to ... alone ...` | Every feature names another channel or measures a shape; check the channel names against the feature names |
| `There are only ... metacells, so every fit uses all of them ...` (warning) | The fits differ only in their starts; lower `coverage_sample_size` and `fit_sample_size` to about 40% of the metacells |
| `Only ... of ... K values survive the drift and support checks ...` | `evalK_recovery_curves.csv` gives the reason for each exclusion; set `archetypes.k` |
| `'archetypes.npc' gives ... but its npc.txt holds ...` | The weights were built with another `npc`; rerun from the `npc` step |
| `No K given and no k.txt in ...` or `No npc.txt ...` | Run the `selectk` or `npc` step first, or set `archetypes.k` |
| `The process running ... stopped before it finished ...` | Usually out of memory; lower `parallel_channels` |
| `Skipping the crops for ...: ...` (warning) | `archetypes.image_channels` lacks the channel, or SCALLOPS is not installed |
| `No '...' is configured for well ... under 'preprocessing_bywell' ...` | Set `pheno_dir`, `cell_labels_dir` and `nuclei_labels_dir`, described under [Your data](data.md) |
| `... cells shown for ... have no outline in the cell label image ...` (warning) | The label images are not the segmentation the objects table came from, or `cell_column` does not hold the label values |
