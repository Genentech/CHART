# Effects

`chart effects` tests every perturbation against the controls on every archetype of each channel, using the archetype weights from `chart archetypes`. Once every channel has been tested, the p-values of all channels are corrected together, and the result, `effects/pooled.parquet`, is the input to `chart programs`.

## Usage

> [!IMPORTANT]
> Run `chart archetypes` through its last step, `crops`, before testing. Inspect each channel's crops, under `archetypes/<channel>/crops/` in `local_output_dir`, and list the archetypes that reflect segmentation errors or staining artefacts under `archetypes.archetypes_to_remove`. These archetypes are not tested, and since all tests are corrected together, removing one also changes the q-values of the rest.

```yaml
archetypes:
  archetypes_to_remove:
    seg_err: [TOM20_Archetype1]
    staining_artifacts: [TOM20_Archetype4]
```

The category names are free; only the archetype names, `<channel>_ArchetypeN`, are read. Then:

```
chart effects --config my_screen.yaml --all-channels
chart effects --config my_screen.yaml --channel TOM20 CALR
```

Channels can be tested in separate runs, for instance as separate cluster jobs. Each run ends by pooling, which happens only once every channel in `archetypes.channels` has a table, from that run or an earlier one; until then it is skipped with a warning. A channel that fails in a run blocks pooling in that run, since any table of its on disk is from an earlier one. `--continue-on-error` carries on with the remaining channels after a failure.

Perturbations are distributed over worker processes, by default one per core available to the run, which under SLURM is the job's allocation. `effects.n_jobs` sets their number for this stage, the top-level `n_jobs` for every stage, and `--n-jobs` overrides both.

## Outputs

All tables are written to `effects/` under `data_output_dir`.

| File | Rows | Contents |
|---|---|---|
| `<channel>.parquet` | perturbation × archetype | effect size, p-value and cell counts |
| `<channel>.json` | | weights read, test settings, archetypes removed, perturbations not tested |
| `pooled.parquet` | perturbation × archetype | every channel's rows, with q-values |
| `pooled.json` | | correction method, number of tests, tables pooled |

| Column | Contents |
|---|---|
| `Gene` | the perturbation |
| `Archetype` | `<channel>_ArchetypeN` |
| `w1` | Wasserstein-1 distance between the perturbation's and the controls' distributions of the archetype's CLR-transformed weight |
| `signed_w1` | `w1`, positive when the perturbation's median exceeds the controls' |
| `w1_trimmed` | `w1` after discarding the outer `trimmed_alpha` of each distribution at both ends |
| `pval` | permutation p-value for `w1` |
| `qval` | Benjamini–Hochberg q-value, in `pooled.parquet` only |
| `n_gene`, `n_ctrl` | perturbation and control cells tested |
| `n_wells` | wells in which the perturbation appears |

`w1` and `signed_w1` are in centred log-ratio (CLR) units. `chart programs` uses `signed_w1` and `qval`; `w1_trimmed` indicates how much of an effect is carried by the tails.

## Configuration

```yaml
effects:
  outputs_dir: effects/
  n_perm: 1000          # permutations per test
  trimmed_alpha: 0.02   # tail fraction discarded for w1_trimmed
  seed: 42
  # n_jobs: -1          # worker processes; the top-level n_jobs unless set
```

The values shown are the defaults, and those of the published analysis. The stage also reads `archetypes.channels`, `archetypes.archetypes_to_remove`, and the controls defined by `schema.control_gene` and `schema.control_prefix`.

A permutation p-value cannot fall below $1/(n_\mathrm{perm} + 1)$, and since Benjamini–Hochberg never lowers a p-value, neither can a q-value. With 1000 permutations the floor is about $10^{-3}$, which leaves room below the default `programs.qval_threshold` of 0.01; a stricter threshold needs more permutations.

## Method

**Log-ratio transform.** Each cell's weights in a channel are smoothed away from zero, $(w + \alpha)/(1 + K\alpha)$ with $\alpha = 10^{-6}$ and $K$ the channel's number of archetypes, log-transformed, and centred on the cell's mean log weight. All of the channel's archetypes enter the transform, including those removed from testing.

**Test.** Controls are the cells whose perturbation is `schema.control_gene` or starts with `schema.control_prefix`. Each perturbation is compared with the control cells pooled from the wells in which it appears. For each archetype, `w1` is the exact one-dimensional Wasserstein-1 distance between the two empirical distributions of the archetype's CLR value, and `signed_w1` carries the sign of the difference in medians. The p-value is $(b + 1)/(n_\mathrm{perm} + 1)$, where $b$ is the number of permutations, shuffling perturbation and control labels within each well, whose distance is at least the observed one. The distance responds to any difference between the two distributions, not only a shift in location, and in either direction. Permutations are seeded from `seed`, so a rerun on the same inputs reproduces the p-values exactly.

**Correction.** The p-values of every channel in `archetypes.channels` are corrected together by Benjamini–Hochberg, over every perturbation × archetype test.

## Troubleshooting

| Message | Cause and remedy |
|---|---|
| `No archetype weights for ... at ...; run the weights step` | `chart archetypes` has not produced weights for that channel with the configured `k` and parameters |
| `None of the ... cells for ... carry a control guide` | No perturbation matches `schema.control_gene` or `schema.control_prefix` |
| `'archetypes.archetypes_to_remove' names ..., but ... has only ...` | An archetype listed for removal does not exist in that channel |
| `... genes have no controls in their wells and were not tested` (warning) | Those perturbations appear only in wells without control cells; they are listed under `skipped_genes` in `<channel>.json` |
| `Nothing in 'archetypes.archetypes_to_remove' belongs to ...` (warning) | No archetype of that channel is excluded; inspect its crops and list artefactual ones before testing |
| `Not pooling the q-values: ...` (warning) | A channel in `archetypes.channels` has no table yet, or failed in this run; test it, and pooling follows |
| `The pooled q-values cover only the channels in 'archetypes.channels' ...` (warning) | A channel was tested that `archetypes.channels` does not list; its table is written but not pooled |
