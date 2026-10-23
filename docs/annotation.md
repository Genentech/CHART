# Annotation

`chart annotation` describes each archetype from its crops with a multimodal language model. For every archetype it writes a morphological summary, three proposed names and a functional hypothesis, and collects them with the images in one HTML dashboard per channel. The stage is optional: nothing downstream reads its output.

It reads the grids drawn by the `crops` step of `chart archetypes`, leaving out the archetypes listed under `archetypes.archetypes_to_remove`. The packages and credentials it needs are described under [Installation](installation.md).

## Usage

Each channel needs one or more normal archetypes, chosen from the crops, against which every archetype is compared:

```yaml
annotation:
  normal_archetypes:
    TOM20: [10]
    FAK: [5, 8, 11]
```

```
chart annotation --config my_screen.yaml --all-channels
chart annotation --config my_screen.yaml --channel TOM20 --steps annotate baseline
```

| Step | Output |
|---|---|
| `annotate` | An observational description of each of the archetype's first two grids |
| `baseline` | The archetype's first grid compared with the normal archetypes' |
| `synthesize` | A summary from the first grid's description and the baseline note; three names and a functional hypothesis |
| `refine` | Summaries that read alike, contrasted pairwise and rewritten |
| `dashboard` | The final reports beside their grids, in one HTML file |

The steps run in order, each reading the previous step's output. `--resume` keeps the descriptions `annotate` has already written, so an interrupted run continues where it stopped. `--continue-on-error` carries on with the remaining channels after a failure. `dashboard` on its own needs no model or credentials.

> [!NOTE]
> Calls are made one at a time. For a channel with $K$ archetypes, `annotate` makes up to $2K$ image calls, `baseline` $K$, `synthesize` $2K$ text calls, and `refine` two image calls per similar pair and one rewrite per archetype in a pair. `refine` compares summaries with the sentence-transformers model `all-MiniLM-L6-v2`, downloaded from Hugging Face on first use; on a machine without internet access, download it beforehand.

## Outputs

All files are written to `annotation/<channel>/` under `local_output_dir`. Archetype numbers are those of the crops, `<channel>_Archetype<N>`.

| File | Contents |
|---|---|
| `archetype_annot/archetype<N>_grid_<g>_annotation.md` | Description of one grid |
| `contrastive_baselines/contrastive_baseline_notes.json` | Each archetype against the normal archetypes; `contrastive_baseline_report.md` holds the same notes |
| `archetype_reviews/archetype<N>_synthesis.md` | Summary, names and hypothesis |
| `archetype_reviews/archetype<N>_contrast_vs<M>.md` | Traits distinguishing archetype N from M |
| `archetype_reviews/archetype<N>_final.md` | Final summary, names, hypothesis, and the archetypes it was contrasted with |
| `archetype_reviews/combined_archetype_synthesis.md`, `combined_archetype_final.md` | Every archetype's report in one file |
| `archetype_dashboard.html` | The final reports with their grids embedded, readable anywhere |

## Configuration

```yaml
annotation:
  reports_dir: annotation/       # under local_output_dir
  normal_archetypes: {}          # per channel, by archetype number
  model: gpt-5.5
  similarity_threshold: 0.08     # cosine distance below which summaries are contrasted
  max_pairs_per_archetype: 0     # closest partners kept per archetype; 0 keeps all
```

The values shown are the defaults. `normal_archetypes` has to be set for every channel before `baseline` runs. The stage also reads `archetypes.channels`, `archetypes.archetypes_to_remove`, and `archetypes.reports_dir`, under which the crops are.

## Method

**Grids.** The crops step draws each archetype's most heavily weighted cells, `archetypes.top_cells` (10 by default), five to a grid: the stain in inverted grey, the cell outlined in red and its nucleus in blue. The first two grids of each archetype are described; only the first enters the later steps.

**Observation.** The descriptions, the baseline comparisons and the pairwise contrasts are restricted to visible features: intensity, localisation relative to the two outlines, texture, cell and nuclear size and shape, and artefacts. The model is instructed not to infer biology or to assume the stain's usual pattern. The only interpretive call is the one in `synthesize` that proposes names and a hypothesis, from the first grid's description.

**Baseline.** Each archetype's first grid is sent together with the normal archetypes' first grids in one request, at reduced resolution (longest edge at most 1024 pixels).

**Refinement.** The summaries are embedded with `all-MiniLM-L6-v2`, and each pair closer than `similarity_threshold` in cosine distance is contrasted in both directions, each call showing both first grids. Every archetype in a pair then has its summary rewritten to include what distinguishes it. With `max_pairs_per_archetype` above 0, only each archetype's closest partners are kept. Archetypes in no pair keep their synthesis summary.

**Validation.** Every reply must contain the sections the next step reads; a reply without them is requested once more, then the step fails. Empty and truncated replies are errors, and rate-limited calls are retried with increasing delays. A step therefore either writes every archetype's report or stops.

## Troubleshooting

| Message | Cause and remedy |
|---|---|
| `No top_cells.parquet in ...; run the crops step of chart archetypes for ...` | The channel has no crops; run `chart archetypes --steps crops` for it |
| `top_cells.parquet lists archetype ..., but ... is not in ...; rerun the crops step` | The grid images are incomplete; rerun the crops step |
| `'annotation.normal_archetypes' has to name archetypes of ... that the crops step drew and 'archetypes.archetypes_to_remove' keeps` | No normal archetype is set for the channel, or one named was removed or not drawn |
| `Missing langchain_openai, sentence_transformers. ...` | The annotation extra is not installed |
| `No LLM provider is configured ...` | No credentials in the environment; see [Installation](installation.md) |
| `... model returned no text ...`, `... reply was cut off at the token ceiling ...` | The model spent its token budget; rerun the step, with `--resume` for `annotate` |
| `... reply is missing required section header(s) ...` | The model twice ignored the output format; rerun the step |
