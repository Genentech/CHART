# CHART

**CHART** (**Cellular High-content imaging Archetype Response Toolkit**) translates high-content imaging features into interpretable morphological phenotypes. CHART identifies marker-specific cellular archetypes, annotates representative images using multimodal large language models, and integrates co-regulated archetypes into morphological programs that describe coordinated cellular responses to perturbation. The method and its application are described in [Coelho, Leote et al. (2026)](https://doi.org/10.64898/2026.10.07.757470), *bioRxiv*.

![Overview of the CHART framework](../images/chart_diagram.png)

- **OPS preprocessing** — quality control, filtering, normalization, outlier detection, guide filtering and single-cell/aggregated embeddings.
- **Compartment archetype analysis** — identification of distinctive reference morphological states for each imaged marker.
- **Automated archetype annotation** — contrastive image analysis and standardized reports generated with multimodal language models and user review.
- **Morphological program definition** — integration of co-regulated archetypes into interpretable programs of coordinated cellular change.

Start with [Installation](installation.md) and [Your data](data.md); a quickstart is in the [README](https://github.com/Genentech/CHART#readme).

:::{toctree}
:hidden:

installation
tutorial
data
configuration
preprocessing
archetypes
annotation
effects
programs
Cluster <cluster>
:::

## Citation

If you use CHART in your work, please cite:

> **Archetype analysis of multiplexed antibody imaging reveals interpretable morphological programs in optical pooled screens**
>
> P. P. Coelho, A. C. Leote, H. Yao, P. Wu, T. Kudo, H. Wang, V. Ermakov, A. Saboowala, T. Makino, R. Littman, R. Lopez, J. Kim, T. Hafner, O. Rozenblatt-Rosen, D. Richmond, L. Garraway, A. Singh, J. Marioni, A. Regev
>
> _bioRxiv_ 2026. doi: [10.64898/2026.10.07.757470](https://doi.org/10.64898/2026.10.07.757470)
