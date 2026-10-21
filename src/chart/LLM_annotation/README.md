# Archetype annotation pipeline

Turns **archetypes** into written, reviewable
descriptions using a vision LLM.

Given a folder of grid images — one image per archetype, each showing five
representative cell crops — the pipeline produces, per archetype: an
observational annotation, a comparison against a designated "normal" archetype,
a summary with proposed names and a functional hypothesis, a refined summary
that separates it from look-alike archetypes, and an HTML dashboard putting the
images and the text side by side.

```
grid images ──▶ 1 annotate ──▶ 2 baseline ──▶ 3 synthesize ──▶ 4 refine ──▶ 5 dashboard ──▶ 6 verify
                  observe       vs normal      name + explain    disambiguate    publish       check
```

---

## 1. Install

```bash
conda create -n archetype-annot python=3.11
conda activate archetype-annot
pip install -r requirements.txt
```

`sentence-transformers` pulls in torch and downloads `all-MiniLM-L6-v2` on
first use (step 4 only).

Point the runner at your interpreter if it is not on `PATH`:

```bash
export PYTHON=/path/to/envs/archetype-annot/bin/python
```

## 2. Credentials

Configure **either** provider — the pipeline is identical on both.

```bash
# LiteLLM gateway (OpenAI-compatible)
export LITELLM_API_KEY="sk-..."
export LITELLM_BASE_URL="https://<gateway-host>"     # '/v1' appended if absent

# or Azure OpenAI
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2025-03-01-preview"          # optional
export AZURE_DEPLOYMENT_MAP='{"gpt-5.5": "gpt-5-5-deploy"}'   # optional
```

Nothing is hardcoded; every credential comes from the environment.
`run_pipeline.sh` also sources `~/.litellm_env` and `./.litellm_env` if they
exist (override with `LLM_ENV_FILE`), so runs work from cron and non-interactive
shells.

`PROVIDER=auto` (the default) uses the gateway when its variables are set and
falls back to Azure. `PROVIDER=gateway` or `PROVIDER=azure` pins one — pinning
an unconfigured provider is an error, never a silent fallback. Every step also
accepts `--provider`, and `LLM_PROVIDER` is the environment equivalent. The
resolved provider is printed at preflight and by each step:

```
🔌 Provider: azure https://res.openai.azure.com/ (deployment=gpt-5-5-deploy, api_version=2025-03-01-preview)
```

## 3. Prepare the inputs

### Image files

All grid images for one stain live in a single folder, named:

```
archetype<i>_<STAIN_TAG>_grid_<n>.png

archetype1_PRPS6_grid_1.png
archetype1_PRPS6_grid_2.png
archetype2_PRPS6_grid_1.png
...
```

- `<i>` — archetype id (integer, as produced by the clustering).
- `<STAIN_TAG>` — the stain as it appears in the filename. The stain name given
  on the command line is matched case-insensitively; extra spellings are
  declared in `STAIN_FILE_TAG_ALIASES` in [utils.py](utils.py) (for example
  `TUBULIN` also matches `B-TUBULIN`).
- `<n>` — grid index. **`grid_1` is required for every archetype**; it is what
  steps 1–4 read. `grid_2` is optional: annotated and shown in the dashboard
  when present, rendered as a placeholder when not.

Each grid image should be one row of five cell crops, rendered the way the
prompts describe (see `IMAGE_CONVENTIONS` in [promptlib.py](promptlib.py)):

- grayscale with `cmap='gray_r'` — **darker = stronger fluorescence**;
- red outline = cell boundary, blue outline = nuclear boundary;
- crop pixel dimensions visible, so relative cell and nuclear size can be judged.

Per-well intensity scaling is expected: crops are comparable across wells.

### Register the stain

Add one line to `STAIN_CONFIG` in [run_pipeline.sh](run_pipeline.sh):

```bash
#  "STAIN"="k:normal_archetype_ids:image_folder_suffix"
["PRPS6"]="15:5:PRPS6"
["FAK"]="12:5,11,8:FAK-k12"
```

| Field | Meaning |
|---|---|
| `k` | Number of archetypes the clustering produced. Checked at every step; a run cannot cover 9 of 11 archetypes and call itself done. |
| `normal_archetype_ids` | Archetype id(s) that define the *normal* baseline for step 2, comma-separated. Each must have a `grid_1` image. |
| `image_folder_suffix` | Images are read from `$BASE_PATH/<suffix>`. |

Then set the image root:

```bash
export BASE_PATH=/path/to/archetype/images     
```

## 4. Run it

```bash
./run_pipeline.sh                          # every stain in STAIN_CONFIG
./run_pipeline.sh PRPS6                    # one stain
./run_pipeline.sh PRPS6 TUBULIN            # several
MODEL=gpt-5 ./run_pipeline.sh PRPS6        # a different model
PROVIDER=azure ./run_pipeline.sh PRPS6     # pin the provider
OUT_ROOT=/data/results ./run_pipeline.sh PRPS6   # write results elsewhere
```

Preflight checks the interpreter, the packages, the credentials and the image
path before the first API call. A stain stops at its first failed step; other
stains continue; the script exits non-zero if any stain failed. Timings per step
are appended to `<review_dir>/workflow_timing.log`.

Every step is also a standalone CLI — `python step3_synthesize.py --help` — so a
single step can be re-run against an existing run folder.

## 5. What each step does

Paths below are relative to `OUT_ROOT` (default: this folder), with `{STAIN}`
the stain name and `{i}` the archetype id.

### Step 1 — annotate · [step1_annotate.py](step1_annotate.py)

One vision call per grid image producing a **purely observational** report:
per-cell intensity (on a fixed six-point scale), localisation relative to the
red/blue outlines, texture, cell and nuclear shape, artifacts; then the traits
shared across the five cells. No biology, no marker priors — the model is told
explicitly not to describe what the stain *usually* does.

The whole stain runs in one process so the archetype count can be checked
against `k`. Filenames are deterministic, so re-runs overwrite instead of
accumulating; `--resume` reuses existing non-empty reports.

- **in** grid images · **out** `archetype_annot_{STAIN}/archetype{i}_grid_{n}_annotation.md`
- **needs** `### Cell Observations` and `### Shared Traits` in every reply

### Step 2 — contrastive baseline · [step2_contrastive_baselines.py](step2_contrastive_baselines.py)

For each archetype, sends **its grid image together with the normal
archetype's grid image(s) in one request** and records how the two differ. This
is what later lets step 3 say "nuclei larger than baseline" rather than just
"large nuclei".

The comparison is image-to-image by design. Precomputing one written "normal
baseline" paragraph and comparing everything against that text was tried and is
worse — a paragraph is lossy next to showing the model the actual normal grids
each time. Each call re-forming its own impression of normal is intended.

- **in** grid images, `--normal <ids>` · **out** `contrastive_baselines_{STAIN}/contrastive_baseline_notes.json` (+ a readable `.md`)
- The JSON is re-read after writing, and its archetype count becomes the
  `effective_k` used by steps 3–6.

### Step 3 — synthesize, name, hypothesise · [step3_synthesize.py](step3_synthesize.py)

Text only. Per archetype, two calls: the grid-1 annotation plus the baseline
note become a ~200-word **Summary**; a second call proposes three descriptive
**names** and one **functional hypothesis**. This is the only step allowed to
interpret biology.

- **in** step 1 reports + step 2 JSON · **out** `archetype_reviews_{STAIN}/archetype{i}_synthesis.md`, plus a combined file
- Each report is re-parsed as it is written, so an unparseable report fails here
  rather than in step 4.

### Step 4 — contrastive refinement · [step4_refine.py](step4_refine.py)

Summaries within one stain read alike, which makes archetypes hard to tell
apart. This step embeds every step 3 summary (`all-MiniLM-L6-v2`), flags pairs
closer than `--similarity_threshold` in cosine distance, sends each flagged pair
as two images **in both directions** asking only for distinguishing traits, then
rewrites each summary to fold those distinctions in. Archetypes with no similar
partner keep their step 3 summary and say so in the report.

`--similarity_threshold` (default
0.08 in `run_pipeline.sh`) and `--max_pairs_per_archetype` (0 = no cap) are the
levers; capping logs exactly which pairs it dropped.

- **out** `archetype{i}_contrast_vs{j}.md`, **`archetype{i}_final.md`**, `combined_archetype_final.md`
- `archetype{i}_final.md` = refined Summary + Proposed Names and Functional
  Hypothesis

### Step 5 — dashboard · [step5_build_dashboard.py](step5_build_dashboard.py)

Renders the final reports next to their grid images as one standalone HTML file
with the images base64-embedded, so it can be emailed or opened anywhere.
Refuses a report with no Summary, or a missing required grid image
(`--allow_missing_images` overrides).

- **out** `dashboards_from_final/archetype_dashboard_{STAIN}.html`

### Step 6 — verify · [step6_verify_run.py](step6_verify_run.py)

Writes nothing. Walks every artifact the stain should have produced — step 1
annotations, step 2 notes, step 3 sections, step 4 finals and contrast notes,
step 5 dashboard — and exits non-zero listing **all** problems if anything is
missing, empty or malformed.

## 6. Output layout

```
OUT_ROOT/
├── archetype_annot_{STAIN}/          step 1  archetype{i}_grid_{n}_annotation.md
├── contrastive_baselines_{STAIN}/    step 2  contrastive_baseline_notes.json, .md
├── archetype_reviews_{STAIN}/        step 3  archetype{i}_synthesis.md
│                                     step 4  archetype{i}_final.md   ← the deliverable
│                                             archetype{i}_contrast_vs{j}.md
│                                             combined_archetype_final.md
│                                             workflow_timing.log
└── dashboards_from_final/            step 5  archetype_dashboard_{STAIN}.html
```

## 7. Configuration reference

Set per step in `run_pipeline.sh`, or passed directly to a step's CLI.

| Knob | Where | Notes |
|---|---|---|
| `MODEL` | env | `gpt-5.5` (the runner's default) and `gpt-5` (each step's own default when called directly) are registered reasoning models; `gpt-4o` is registered as a temperature model. Unregistered names still run: `gpt-5*` and `o1/o3/o4*` are assumed to be reasoning models, everything else temperature, and the guess is logged. |
| `--max_completion_tokens` | per step | **Reasoning tokens are billed against this.** A high-effort step with a small budget can spend the entire budget thinking and return an empty string. 8000 for steps 1–3, 16000 for step 4. |
| `--reasoning_effort` | per step | `low` for the observational steps, `medium`/`high` for synthesis and refinement. Ignored by temperature models. |
| `--temperature` | per step | Used by temperature models only; ignored by GPT-5 style models. |
| `--similarity_threshold` | step 4 | Cosine distance below which two summaries count as too similar. Higher = more pairs = more cost. |
| `--max_pairs_per_archetype` | step 4 | Keep only the N closest partners per archetype. 0 = no cap. |
| `--expected_archetypes` | all steps | The count check. 0 disables it. |
| `--grids` / `--required_grids` | steps 1, 5 | Which grid indices to process, and which must exist. |

Azure addresses *deployments*, not model names, so a deployment named
differently from the model needs an `AZURE_DEPLOYMENT_MAP` entry.

## 8. Module map

| File | Role |
|---|---|
| [run_pipeline.sh](run_pipeline.sh) | Stain table, per-step budgets, preflight, orchestration |
| [step1_annotate.py](step1_annotate.py) … [step6_verify_run.py](step6_verify_run.py) | The six steps, each a standalone CLI |
| [promptlib.py](promptlib.py) | Every prompt the pipeline sends |
| [llm_core.py](llm_core.py) | Provider resolution, model registry, retries, response validation, cost accounting |
| [image_utils.py](image_utils.py) | Image loading, downsampling, base64 blocks, the vision call |
| [utils.py](utils.py) | Filename conventions, archetype discovery, report parsing, summary similarity |
