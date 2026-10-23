# CHART

**CHART** (**Cellular High-content imaging Archetype Response Toolkit**) translates high-content imaging features into interpretable morphological phenotypes. CHART identifies marker-specific cellular archetypes, annotates representative images using multimodal large language models, and integrates co-regulated archetypes into morphological programs that describe coordinated cellular responses to perturbation.

![Overview of the CHART framework](images/chart_diagram.png)

## Components

- **OPS preprocessing** — quality control, filtering, normalization, outlier detection, guide filtering and single-cell/aggregated embeddings.
- **Compartment archetype analysis** — identification of distinctive reference morphological states for each imaged marker.
- **Automated archetype annotation** — contrastive image analysis and standardized reports generated with multimodal language models and user review.
- **Morphological program definition** — integration of co-regulated archetypes into interpretable programs of coordinated cellular change.

## Documentation

The [documentation](https://genentech.github.io/CHART/) describes the input data, each stage's usage, outputs and method, the configuration, and running on a cluster.

## Installation

```
pip install .
```

The archetype analysis also needs R and the packages in `environment.yml`:

```
mamba env create -f environment.yml
conda activate chart-r
Rscript -e 'install.packages("archetypes", repos="https://cloud.r-project.org")'
```

The optional annotation stage needs `pip install ".[annotation]"` and credentials for a language model in the environment. [Installation](https://genentech.github.io/CHART/installation.html) covers these, scallops for reading images, and remote storage.

## Quickstart

Write a config saying where the input is and where output should go:

```yaml
data_output_dir: /work/screen1/bulk
local_output_dir: /work/screen1/reports

preprocessing_bywell:
  merged_dir: /data/screen1/objects        # where the input parquet lives

archetypes:
  channels: [TOM20]                        # the channels to analyse
  all_channels: [DAPI1, TOM20, Golgin97]   # every channel the table holds
```

Then run the five stages in order:

```
chart preprocessing --config my_screen.yaml --all-wells
chart archetypes --config my_screen.yaml --all-channels
chart annotation --config my_screen.yaml --all-channels   # optional
chart effects --config my_screen.yaml --all-channels
chart programs --config my_screen.yaml
```

Before `chart effects`, inspect each archetype's crops under `archetypes/<channel>/crops/` in `local_output_dir`, and list those reflecting segmentation errors or staining artefacts under `archetypes.archetypes_to_remove`, as described in [Curating archetypes](https://genentech.github.io/CHART/archetypes.html#curating-archetypes). `example_config.yaml` lists every setting, and [Configuration](https://genentech.github.io/CHART/configuration.html) explains which matter most.
