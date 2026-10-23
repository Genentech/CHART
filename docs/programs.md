# Programs

`chart programs` decomposes the perturbation × archetype effect matrix into morphological programs, sparse sets of archetypes across channels that perturbations shift together. Perturbations are grouped into modules by consensus clustering of their program scores, and every cell is scored on the same programs, which links each program back to individual cells and their images.

The stage reads `effects/pooled.parquet` from `chart effects`, the per-channel archetype weights from `chart archetypes`, and the objects table from `chart preprocessing`.

## Usage

```
chart programs --config my_screen.yaml
```

The four steps run in order, each reading the previous step's output; `--steps` selects a subset:

```
chart programs --config my_screen.yaml --steps modules cells
```

| Step | Output |
|---|---|
| `matrix` | Signed effects of the significant perturbations on the significant archetypes |
| `fit` | Program loadings and perturbation scores |
| `modules` | Consensus matrix and module assignments |
| `cells` | Cell scores and example cells per program |

> [!NOTE]
> `modules` refits the decomposition 150 times; for 471 perturbations and 188 archetypes it takes about 20 minutes on 96 cores.

The sparse PCA uses every core available to the run, which under SLURM is the job's allocation; the top-level `n_jobs`, or `--n-jobs`, sets a number instead.

## Outputs

All tables are written to `programs/` under `data_output_dir`.

| File | Rows | Contents |
|---|---|---|
| `matrix.parquet` | perturbation | signed effect on each retained archetype |
| `loadings.parquet` | archetype | loading on each program |
| `scores.parquet` | perturbation | score on each program |
| `consensus.parquet` | perturbation | co-clustering frequency with every other perturbation |
| `modules.parquet` | perturbation | module, nearest other module, margin, core or weak |
| `cell_scores.parquet` | cell | score on each program |
| `top_cells.parquet` | program × rank | example cells with score, purity and bounding box |

Positive loadings mark archetypes in which perturbations scoring high on the program gain weight relative to controls; negative loadings, archetypes in which they lose it. Perturbation and cell scores are both uncentred projections in centred log-ratio (CLR) units. A perturbation scores zero when it does not differ from controls; a cell scores zero when its weights are uniform across every channel's archetypes.

A perturbation's `margin` is its mean consensus with its own module minus its highest mean consensus with any other (`best_other_module`). Perturbations with a margin below 0.15 are `weak`, the rest `core`.

`matrix`, `scores`, `modules` and `cell_scores` each come with a `.json` record of the inputs and settings that produced them.

## Configuration

```yaml
programs:
  outputs_dir: programs/
  k: 18                  # number of programs
  qval_threshold: 0.01   # significance for retaining perturbations and archetypes
  n_modules: 25          # number of modules
  seed: 42
```

The defaults are those of the published analysis, 471 perturbations and 188 archetypes; `k` and `n_modules` should be chosen for the screen at hand.

## Method

**Effect matrix.** `chart effects` measures each perturbation's effect on each archetype as the Wasserstein-1 distance between the distributions of that archetype's CLR-transformed weight in its cells and in the control cells of the same wells, signed by the difference in medians (`signed_w1`). P-values come from permuting labels within wells, and Benjamini–Hochberg q-values are computed jointly across channels. The matrix keeps every perturbation with q < `qval_threshold` on at least one archetype, and every archetype with at least one such perturbation.

**Decomposition.** Sparse PCA (scikit-learn, `k` components, at most 500 iterations, tolerance $10^{-4}$) factorises the matrix; in the published analysis about a quarter of the loadings are non-zero. Each program's sign is fixed so that its largest-magnitude loading is positive. Scores are the ridge projection $S = XL\,(L^\top L + 0.01\,I)^{-1}$, as in scikit-learn's `transform` but without centring $X$, so that zero means no effect.

**Consensus modules.** The decomposition is refitted on 150 random subsamples of 85% of the perturbations (10 seeds × 15 subsamples). Each refit's programs are matched to the reference by Hungarian assignment on the absolute cosine similarity of their loadings and sign-aligned. Each refit's scores are then partitioned 38 ways: Ward and average-linkage hierarchical clustering of the z-scored scores, cut at 12, 14, …, 28 clusters, and Leiden clustering (RB configuration, resolution 0.8, 1.0, 1.2, 1.5, 1.8) of symmetrised nearest-neighbour graphs with 5, 8, 10 and 12 neighbours. The consensus of two perturbations is the fraction of partitions containing both that place them together. Modules are obtained by average-linkage clustering of 1 − consensus, cut into `n_modules` clusters.

**Cell scores.** Each cell's weights are CLR-transformed per channel, as in `chart effects`, concatenated across channels and projected with the same ridge projection. A cell absent from a channel contributes zero there, as if its weights were uniform. For each program, the example cells are the 20 with the highest purity among its 100 top-scoring cells, purity being the cell's percentile on the program minus its mean percentile on the other programs.

## Troubleshooting

| Message | Cause and remedy |
|---|---|
| `No pooled.parquet in ...; run 'chart effects' first` | Pooled effects are written only once every channel in `archetypes.channels` has been tested; run `chart effects` on the rest |
| `No gene has a q-value below ...` | No effect passes `qval_threshold` |
| `... genes were not tested on every archetype kept ...` | Those perturbations' wells contain no control cells in some channel, so they have no effect there |
| `The loadings in ... were fitted on another matrix; rerun the fit step` | The matrix changed after the decomposition; rerun from `fit` |
| `The programs load on ..., which 'archetypes.archetypes_to_remove' now leaves out` | An archetype was excluded after the programs were fitted; rerun `chart effects`, then this stage from `matrix` |
| `The programs load on ..., which the weights do not hold` | The archetypes were refitted, or a channel removed from `archetypes.channels`, since the programs were fitted; rerun from `matrix` |
