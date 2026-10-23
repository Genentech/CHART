# Running on a cluster

Each stage is a single command, so on a cluster it becomes one batch job, or one per well or channel. This page describes how the stages divide into jobs, what they use of a job's cores and memory, and what the compute nodes need. The examples use SLURM.

`data_output_dir` and `local_output_dir` have to be reachable from every node that runs a job, on a shared filesystem or on [remote storage](installation.md).

## Dividing a run into jobs

| Order | Command, each with `--config` | Jobs |
|---|---|---|
| 1 | `chart preprocessing --well <well> --steps qc filter` | One per well |
| 2 | `chart preprocessing --all-wells --steps combine normalize outliers guides` | One, after every job of 1 |
| 3 | `chart archetypes --channel <channel>` | One per channel |
| 4 | `chart annotation --channel <channel>` | Optional; one per channel, after 3 |
| 5 | `chart effects --channel <channel>` | One per channel, after 3 and curation |
| 6 | `chart programs` | One, after every job of 5 |

Each row can also run as a single job, with `--all-wells` or `--all-channels`. Between 3 and 5 the archetypes are curated by hand, as described under [Archetypes](archetypes.md). Each effects job pools the q-values once every channel has a table, so the last to finish writes `pooled.parquet`. Every command exits with a non-zero status when a step fails, so dependent jobs are held back.

A job array runs the per-well steps, and a dependency starts the across-wells steps once all of them have succeeded:

```bash
#!/bin/bash
#SBATCH --array=0-5
wells=(A1 A2 A3 B1 B2 B3)
chart preprocessing --config my_screen.yaml --well "${wells[$SLURM_ARRAY_TASK_ID]}" --steps qc filter
```

```
jobid=$(sbatch --parsable bywell.sh)
sbatch --dependency=afterok:$jobid allwells.sh
```

`chart archetypes --dry-run` prints every command a job would run, without running any and without R, which checks a configuration before submission.

## Cores and memory

`n_jobs: -1`, the default, uses the cores the process may run on, which under SLURM are the job's allocation rather than the node's.

| Stage | Parallelised over | Set by |
|---|---|---|
| `chart preprocessing` | Threads in `normalize`, and the local outlier factor in `guides` | `n_jobs`, `--n-jobs` |
| `chart archetypes` | Channels, as separate processes; the R scripts are not parallelised | `archetypes.parallel_channels`, `--parallel-channels` |
| `chart annotation` | Nothing; one model call at a time | |
| `chart effects` | Perturbations, over worker processes | `effects.n_jobs`, `n_jobs`, `--n-jobs` |
| `chart programs` | The sparse PCA | `n_jobs`, `--n-jobs` |

Memory is the limiting resource for `chart archetypes`. For one channel of 875,000 cells, the `metacells` step took about 5 hours and held 62 GB, and the `weights` step took 10 minutes. A job running several channels at once with `parallel_channels` needs that memory for each, and shares its cores evenly between them. `normalize` holds two copies of the screen's feature matrix as 64-bit floats, about 1.2 GB each for 300,000 cells by 500 features. The `modules` step of `chart programs` took about 20 minutes on 96 cores for 471 perturbations and 188 archetypes.

## Compute nodes

`chart archetypes` needs `Rscript` on `PATH`, so the job script activates the R environment described under [Installation](installation.md); the stage checks that R can load every package before its first step. The `pheno` and `pheno_sbs` components and the `crops` step need scallops on the node.

`chart annotation` needs network access to the model's endpoint, and the credentials in environment variables. SLURM passes the submitting shell's environment to the job by default, so they can be exported before `sbatch`. The `refine` step loads the all-MiniLM-L6-v2 model from Hugging Face on first use; where compute nodes have no internet access, download it beforehand into a Hugging Face cache (`HF_HOME`) that the nodes share.

## Logs

`chart preprocessing` writes its log to `preprocessing/bywell/qc_reports/` under `local_output_dir`, one file per run named after its well, the number of wells, or `all_wells`, and a timestamp, as well as to standard output. The other stages log to standard error, which SLURM writes to the job's output file, together with the output of the R scripts. With `parallel_channels`, each channel instead logs to `archetypes_log_<timestamp>.txt` in `archetypes/<channel>/` under `local_output_dir`, and the job's output records when each channel starts and ends.
