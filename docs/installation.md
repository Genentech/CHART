# Installation

CHART is a Python package (Python ≥ 3.9) that installs the `chart` command. Some parts of the pipeline need more than the package itself.

| Component | Needed for | Install |
|---|---|---|
| CHART | every stage | `environment.yml` |
| R 4.5 and its packages | `chart archetypes`, every step but `pca` and `crops` | `environment.yml` |
| SCALLOPS | the `crops` step of `chart archetypes`; the `pheno` and `pheno_sbs` QC components | `environment.yml` |
| annotation extra | `chart annotation` | `pip install ".[annotation]"` |
| fsspec and a filesystem driver | inputs or outputs on remote storage | `pip install fsspec s3fs` for S3 |

## CHART

`environment.yml` describes a conda environment, `chart`, with Python 3.12, CHART, SCALLOPS, and R 4.5 with the packages the archetype scripts load. The environment installs CHART from the repository, so the commands run from its root. The `archetypes` R package is on CRAN but not conda-forge, so it is installed separately:

```
git clone https://github.com/Genentech/CHART.git
cd CHART
mamba env create -f environment.yml
conda activate chart
Rscript -e 'install.packages("archetypes", repos="https://cloud.r-project.org")'
```

Without conda, `pip install .` installs CHART alone into an existing Python environment; R and SCALLOPS are then installed separately.

## R

All steps of `chart archetypes` except `pca` and `crops` are R scripts, which the stage runs with `Rscript`. `Rscript` must be on `PATH` when `chart archetypes` runs, as it is with `chart` activated. Before its first step, the stage checks that R can load every package the scripts need, and names any it cannot. The compiled packages, `r-supercell` in particular, are built for one R minor version, which is why the environment fixes R to 4.5.

## SCALLOPS

[SCALLOPS](https://github.com/Genentech/scallops), described in [Gould et al. (2026)](https://doi.org/10.64898/2026.05.22.727250), *bioRxiv*, reads the registered images. Without it, CHART skips the components that need images, with a warning: the crops of archetypal cells, and the phenotype-correlation QC components. An explicit request for one of them, `--steps crops` or `--components pheno`, fails instead. SCALLOPS is part of the `chart` environment; installed separately, with `pip install scallops`, it requires Python ≥ 3.12.

## Annotation

`chart annotation` describes archetypes from their crops with a multimodal language model. Its packages are an optional extra:

```
pip install ".[annotation]"
```

The model is set by `annotation.model` in the config, `gpt-5.5` by default. Credentials are read from the environment only. Two providers are supported:

| Provider | Variables |
|---|---|
| OpenAI-compatible gateway, such as LiteLLM | `LITELLM_API_KEY` and `LITELLM_BASE_URL`; `OPENAI_API_KEY` and `OPENAI_BASE_URL` are also accepted, and `OPENAI_API_KEY` alone addresses OpenAI itself |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT`; optionally `AZURE_OPENAI_API_VERSION`, and `AZURE_DEPLOYMENT_MAP` where deployment names differ from model names, e.g. `'{"gpt-5.5": "gpt-5-5-deploy"}'` |

`LLM_PROVIDER` selects `gateway`, `azure` or `auto`; `auto`, the default, uses the gateway when its variables are set and Azure otherwise.

## Remote storage

Paths in the config may point to remote storage, such as `s3://` URLs. Reading and writing them needs `fsspec` and the driver for the filesystem, `s3fs` for S3, which are not installed with CHART.
