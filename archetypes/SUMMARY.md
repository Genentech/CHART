# Archetype Analysis Pipeline Summary

This folder implements per-channel archetype analysis for CP2.0 (Cell Painting 2.0)
imaging data. The pipeline identifies morphological archetypes from single-cell
fluorescence features, independently for each imaging channel.

## Overview

```
Cell features (per channel, from CellProfiler)
  → PCA embedding (centered, per channel)
    → SuperCell metacells (coarse-graining)
      → Robust archetype analysis on metacell means
        → Archetype weights projected back to single cells
          → Downstream perturbation analysis (Wasserstein distances)
```

## Production pipeline

### Step 1: Adaptive metacell construction

**Script:** `metacell_adaptive_npc.R`
**Submission:** `submit_adaptive_npc.sh`

Each channel's cell-level PCA embedding is coarse-grained into metacells using the
[SuperCell](https://github.com/GfellerLab/SuperCell) R package
(`SCimplify_from_embedding`). The number of PCs (`npc`) is chosen **adaptively per
channel** using an elbow method on the per-PC variance curve (clamped to 10–30).

Fixed parameters across all channels:
- **gamma** × **k.knn** grid: {10, 30} × {5, 10} = 4 combinations per channel
  (listed in `params_adaptive.txt`)

The elbow method (`pick_npc_by_elbow`) finds the PC index with the maximum distance
to the chord connecting the first and last points on a log-variance curve.

Metacell outputs per channel live in `tmp_archetype_outputs/<channel>/`:
- `supercells-<ch>_gamma<G>_knn<K>_npc<P>.rds` — SuperCell object with membership
- `supercells-<ch>_gamma<G>_knn<K>_npc<P>.parquet` — mean PC coordinates per metacell
- `npc_elbow_gamma<G>_knn<K>.pdf` — diagnostic plot of elbow selection

### Step 2: Metacell partition QC

**Script:** `eval_spaces_perchannel.R`
**Submission:** `run_eval_spaces.sh`

Before running archetypes, each metacell partition is evaluated with K-agnostic
quality metrics:
- **Metacell size distribution** — fraction of very small metacells (< 5 or < 10 cells)
- **Within-dispersion** — average within-metacell variance (coherence)
- **Neighbor separation** — ratio of nearest-centroid distance to intra-metacell
  RMS radius (compactness vs. separation)

Results saved as `space_evals.csv` per channel. Used to sanity-check
parameter combinations before the expensive archetype step.

### Step 3: Robust archetype analysis

**Script:** `robustAA_tests/robust_archetyping.R`
**Submission:** `robustAA_tests/submit_all_channels_AA.sh`

Core function: `run_one_space(X)`. For each K in 6–20:

1. **Farthest-point sampling (FPS):** Select a core set C of 12,000 metacells that
   geometrically covers the embedding space.

2. **kNN rarity scoring:** Compute the distance to each metacell's 30th nearest
   neighbor. High values = rare/peripheral points.

3. **Multiple independent fits (R = 8 runs):** Each run draws a different subsample
   of 12,000 metacells (70% from core set C, 30% rarity-weighted) and fits
   archetypes with 10 random restarts (`stepArchetypes`, best model kept).

4. **Held-out evaluation:** A fixed eval set of 8,000 metacells (50% from C, 50%
   rarity-weighted) is used to score each run's archetypes via constrained simplex
   projection (QP: weights ≥ 0, sum to 1). The run with the lowest eval-RSS
   becomes the reference.

5. **Stability diagnostics per K:**
   - **Drift:** Archetypes from each run are matched to the reference via the
     Hungarian algorithm. Low median/p90 Euclidean distance = stable positions.
   - **Support:** Minimum number of eval metacells with weight > 0.3 on any single
     archetype. Guards against spurious/empty archetypes.

Output: `archetype_robust_runs_gamma<G>_knn<K>_npc<P>.rds` containing per-K results
with the reference archetype matrix `Z_ref`, eval-RSS distribution, drift, and
support statistics.

### Step 4: K selection via full-space recovery

**Script:** `robustAA_tests/test_evalK_onfullPC.R`
**Submission:** `robustAA_tests/run_evalK.sh`

Tests whether archetypes found in the `npc`-dimensional space can reconstruct the
**full PC space**. For each K and parameter combination:

1. Project eval metacells onto K archetypes (simplex weights via QP)
2. Learn full-dimensional archetype coordinates via ridge regression
3. Compute reconstruction RMSE (unweighted, size-weighted, and 10%-trimmed)

Produces recovery curves (RMSE vs K) saved as `evalK_recovery_curves.csv` and
`evalK_recovery_plot.png` per channel, used to pick the final K.

### Step 5: Extract single-cell archetype weights

**Script:** `RobustArchetyping.Rmd`

For the chosen K per channel:

1. Load `Z_ref` from the robust runs output
2. Refit archetypes on the full metacell space with 20 restarts
3. `predict()` onto the single-cell PCA embedding to get per-cell archetype weights

Output: `robust_archweights-<ch>_k<K>_gamma<G>_knn<K>_npc<P>.parquet`
— one row per cell with K archetype weight columns plus metadata (Well, Label,
Guide, Gene).


> **Quick-start recommendation:** If you want to skip the parameter grid and run a
> single configuration, **gamma = 10, knn = 10** was a consistently good choice
> across channels.

### Downstream: perturbation analysis

**Script:** `robustAA_volcanos/robustAA_volcanos.Rmd`

Wasserstein distances quantify how genetic perturbations shift cells along each
archetype axis. Results in `wasserstein_results_long.parquet`. Volcano plots and
cross-archetype heatmaps are used to identify perturbation-archetype associations.
---

## Key R packages

- `SuperCell` — metacell construction from embeddings
- `archetypes` — archetype analysis (stepArchetypes, bestModel, predict)
- `quadprog` — constrained simplex projection (QP)
- `clue` — Hungarian matching for archetype drift measurement
- `FNN` — fast kNN for rarity scoring and partition QC
- `arrow` — parquet I/O

## Files

### Pipeline scripts (in order of execution)

```
metacell_adaptive_npc.R              # Step 1: adaptive metacell construction
submit_adaptive_npc.sh               # Step 1: SLURM submission wrapper
eval_spaces_perchannel.R             # Step 2: metacell partition QC
run_eval_spaces.sh                   # Step 2: submission wrapper
robustAA_tests/robust_archetyping.R  # Step 3: robust archetype analysis
robustAA_tests/submit_all_channels_AA.sh  # Step 3: SLURM submission wrapper
robustAA_tests/test_evalK_onfullPC.R # Step 4: full-space recovery / K selection
robustAA_tests/run_evalK.sh          # Step 4: submission wrapper
RobustArchetyping.Rmd               # Step 5: extract single-cell weights
```

### Configuration files

```
channels_subset.txt                  # list of 17 channel names
params_adaptive.txt                  # gamma × knn grid (4 lines)
```

### Input data (per channel)

One file per channel containing the cell-level PCA embedding with metadata
columns (Well, Label, Guide, Gene). These are produced upstream by the
single-channel CellProfiler → PCA pipeline.

### R environment

```
renv.lock                            # pinned package versions
renv/                                # renv library cache (or restore from lockfile)
```

## Execution environment

R environment managed with `renv` (lockfile: `renv.lock`). Jobs run on SLURM
(80–200 GB RAM, 12 CPUs, up to 1 day per job). Each channel × parameter combination
is a separate array task.
