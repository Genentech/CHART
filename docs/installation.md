# Installation

CHART is a Python package (Python ≥ 3.9) that installs the `chart` command. Some parts of the pipeline need more than the package itself.

| Component | Needed for | Install |
|---|---|---|
| CHART | every stage | `pip install .` |
| R 4.5 and its packages | `chart archetypes`, every step but `pca` and `crops` | `environment.yml` |
| scallops | the `crops` step of `chart archetypes`; the `pheno` and `pheno_sbs` QC components | `pip install scallops` |
| annotation extra | `chart annotation` | `pip install ".[annotation]"` |
| fsspec and a filesystem driver | inputs or outputs on remote storage | `pip install fsspec s3fs` for S3 |

## CHART

```
git clone https://github.com/Genentech/CHART.git
cd CHART
pip install .
```

## R

All steps of `chart archetypes` except `pca` and `crops` are R scripts, which the stage runs with `Rscript`. `environment.yml` describes a conda environment, `chart-r`, with R 4.5 and the packages the scripts load. The `archetypes` package is on CRAN but not conda-forge, so it is installed separately:

```
mamba env create -f environment.yml
conda activate chart-r
Rscript -e 'install.packages("archetypes", repos="https://cloud.r-project.org")'
```

`Rscript` must be on `PATH` when `chart archetypes` runs, for instance with `chart-r` activated alongside the Python environment holding CHART. Before its first step, the stage checks that R can load every package the scripts need, and names any it cannot. The compiled packages, `r-supercell` in particular, are built for one R minor version, which is why the environment fixes R to 4.5.

## scallops

[scallops](https://github.com/Genentech/scallops) reads the registered images. Without it, CHART skips the components that need images, with a warning: the crops of archetypal cells, and the phenotype-correlation QC components. An explicit request for one of them, `--steps crops` or `--components pheno`, fails instead. scallops requires Python ≥ 3.12.

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
