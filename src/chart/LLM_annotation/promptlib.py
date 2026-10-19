# promptlib.py
"""Every prompt the pipeline sends, in one place.

Steps 1, 2 and 4 are purely observational - no biology, no marker priors. The
only interpretive step is step 3's names + hypothesis call, which works from
the observations alone.

Section headers below are contracts. Downstream steps parse for them and
llm_core.invoke_text() rejects replies that omit them, so change a header here
only together with the parser that reads it (see utils.SECTION_*).
"""

# ---------------------------------------------------------------------------
# Shared blocks
# ---------------------------------------------------------------------------

IMAGE_CONVENTIONS = """
- Grayscale image using `cmap='gray_r'`: darker areas indicate stronger fluorescence.
- Red outline: cell boundary.
- Blue outline: nuclear boundary.
- Each crop includes X x Y pixel dimensions that help indicate relative cell and nuclear size.
""".strip()

NO_BIOLOGY_RULES = """
- Do not infer biology, mechanism, disease state, treatment, or function.
- Use precise scientific observation language appropriate for immunofluorescence images.
- Describe visible features only. If a difference is uncertain, say it is uncertain.
""".strip()


# ---------------------------------------------------------------------------
# Step 1 - per-image annotation
# ---------------------------------------------------------------------------

def get_annotation_prompt(stain: str) -> str:
    return f"""
You are a microscopy image analysis expert. Describe only what is visibly present in this grayscale fluorescence image. The image shows five cell crops arranged left to right and represents one morphological archetype.

## Image Setup
{IMAGE_CONVENTIONS}
- Stain: {stain}.

## Rules
- Describe visible features only.
- Do not infer biology, mechanism, disease state, or function.
- Do not over-rely on what this stain normally does. These may be abnormal cells, so describe what is actually visible in the image rather than what you expect for the marker.
- Focus on signal intensity, localization, texture, cell shape, nuclear shape, and visible artifacts.
- Texture is important. Always look for whether the signal appears smooth, homogeneous, granular, punctate, speckled, fibrillar, filamentous, reticular, clumped, blotchy, or regionally uneven.
- If background or technical artifacts obscure interpretation, say so clearly.

## Terminology Guidance
- Use standard microscopy and marker-appropriate terminology when it matches what is visibly present.
- Prefer specific morphological language over vague language. For example, if supported by the image, terms such as `filamentous`, `fibrillar`, `reticular`, `punctate`, `clustered`, `polarized`, `perinuclear`, `cortical`, or `diffuse cytoplasmic` are appropriate.
- Keep every statement grounded in the image.

## Boundary Interpretation
- Blue outline marks the nucleus. It appears as a thin closed blue oval/circle.
- Red outline marks the full cell boundary.
- Only evaluate signal that belongs to the outlined target cell.
- Ignore signal outside the red outline when describing that cell. Treat signal outside the red outline as neighboring cells or external/background signal unless clearly stated otherwise.
- `nuclear` means the signal is inside the blue outlined oval/circle.
- `cytoplasmic` means the signal is inside the red outline and outside the blue outline.
- `perinuclear` means the signal is outside the blue outline but immediately adjacent to it, still within the red outline. This is a subtype of cytoplasmic localization.
- Do not call clearly intranuclear signal `perinuclear`.
- If signal is inside the blue outlined oval/circle, describe it as nuclear, not perinuclear.
- If signal is between the blue and red outlines, describe it as cytoplasmic or perinuclear depending on how tightly it hugs the nucleus.
- If both compartments contain signal, say `mixed nuclear and cytoplasmic`.
- If the boundary relationship is unclear, say `unclear` rather than guessing.
- Treat the blue boundary as the hard nuclear mask:
  - signal inside blue = nuclear
  - signal outside blue but inside red = cytoplasmic/perinuclear
  - signal outside red = not part of the target cell
- Do not relabel signal inside the blue circle as `perinuclear` just because it is near the nuclear edge.

## Intensity Scale
Use one of these labels for each cell:
- absent: no convincing signal visible
- very weak: barely detectable signal
- weak: low signal clearly present
- moderate: clearly visible but not dominant
- strong: intense, prominent signal
- very strong: extremely intense / darkest signal in the crop

Important:
- It is valid for a cell to have no signal.
- It is valid for a cell to have only very weak signal.
- Do not upgrade weak signal to moderate.
- Do not downgrade very strong signal to moderate-to-strong.
- Base intensity on the visible grayscale contrast within the crop.

## Signal Vocabulary
When relevant, describe signal as:
- diffuse / uniform
- punctate
- aggregated
- gradient-like / regionally enriched

When relevant, also describe texture explicitly as:
- smooth / homogeneous
- fine granular / coarse granular
- speckled
- fibrillar / filamentous
- reticular / network-like
- clumped / blotchy / irregular

If puncta or aggregates are present, note:
- quantity: few / many / scattered / clustered
- size: small / large
- location: nuclear / cytoplasmic / perinuclear / edge / polar / unclear

## Response Format
### Cell Observations
For each of the 5 cells from left to right, report:
- crop size
- signal presence and intensity using exactly one label from: absent / very weak / weak / moderate / strong / very strong
- signal pattern and localization
- signal texture using specific microscopy language whenever visible
- explicitly state whether the visible signal is:
  - nuclear
  - cytoplasmic
  - perinuclear
  - mixed nuclear and cytoplasmic
  - absent
  - unclear
- cell shape and nuclear shape
- whether signal stays within boundaries
- proximity to neighboring cells
- visible artifacts or background issues

Important for each cell:
- First identify the target cell using the red boundary.
- Then identify the nucleus using the blue closed oval/circle inside that same red boundary.
- Base localization only on the relationship between the signal and those two boundaries.
- Do not use signal from outside the red boundary to describe the target cell.
- Use the blue circle as a hard geometric boundary for the nucleus.
- After intensity and localization, explicitly assess texture before finishing the description.

### Shared Traits
Briefly summarize only the features that are consistent across most or all five cells, using scientifically appropriate morphology terms when supported by the image. Include shared texture descriptors if they are consistent.
""".strip()


# ---------------------------------------------------------------------------
# Step 2 - archetype vs normal baseline
# ---------------------------------------------------------------------------

def build_baseline_comparison_prompt(archetype_id: str, stain: str, normal_ids) -> str:
    normal_label = ", ".join(normal_ids)
    return f"""
You are a microscopy image analysis expert.

You will be shown:
- one candidate archetype image
- one or more designated normal/baseline archetype images

Your task is to compare Archetype {archetype_id} against the normal baseline archetype(s) {normal_label}.

Describe only what is visible in the images.

Image conventions:
- Each image is a 5-cell fluorescence microscopy grid.
{IMAGE_CONVENTIONS}

Rules:
{NO_BIOLOGY_RULES}
- Compare only visible properties.

Evaluate:
1. Relative staining intensity
2. Signal localization and distribution
3. Signal texture or pattern (diffuse, punctate, aggregated, filamentous, patchy, etc.)
4. Cell size/shape and nuclear size/shape
5. Visible artifacts or background differences
6. Whether Archetype {archetype_id} appears similar to, or distinct from, the normal baseline

Output format:
#### Archetype {archetype_id}:
- Size:
- Shape:
- Staining intensity:
- Staining pattern:
- Localization:
- Artifacts or background:
- Relative to normal baseline:
""".strip()


# ---------------------------------------------------------------------------
# Step 3 - synthesis, names, hypothesis
# ---------------------------------------------------------------------------

def build_summary_prompt(archetype_id: str, report_text: str, baseline_note: str) -> str:
    baseline_block = baseline_note.strip() or "*No baseline comparison note available for this archetype.*"
    return f"""
You are an expert in microscopy cell analysis.

You are provided with:
- One report describing a single representative grid for Archetype {archetype_id}
- A baseline comparison note describing how this archetype differs from the designated normal baseline

Your task:

Write a concise (~200 word) summary of the archetype's visible features. Focus on:
   - Fluorescence intensity: strength, consistency, and variability.
   - Localization patterns: nuclear, cytoplasmic, perinuclear, edge-localized, etc.
   - Texture and distribution: puncta, aggregates, gradients, or diffuse patterns if consistently present.
   - Cell and nuclear size: state if they are smaller, larger, or similar compared to baseline when the baseline note supports that comparison.
   - Cell and nuclear shape.
   - Technical artifacts: background noise, signal spillover, uneven staining, or patchy regions.

Use stain-appropriate scientific terminology when supported by the report text. Do not invent features that are not in the report.

Use exactly this format:

### Summary:
...

---

## Grid 1 Report:
{report_text.strip()}

## Baseline Comparison Note:
{baseline_block}
""".strip()


def build_names_prompt(archetype_id: str, report_text: str, stain: str) -> str:
    return f"""
You are an expert in microscopy cell analysis.

You are provided with one report describing a representative grid image for an archetype stained with {stain}.

Your tasks:

1. Propose 3 descriptive names for this archetype.
2. Write one biological hypothesis explaining the likely functional state suggested by the observed morphology and staining pattern.

Do not invent features that are not present in the report.

Use exactly this format:

### Proposed Names:
1. ...
2. ...
3. ...

### Functional Hypothesis:
- ...

---

## Grid 1 Report:
{report_text.strip()}
""".strip()


# ---------------------------------------------------------------------------
# Step 4 - pairwise contrast and refinement
# ---------------------------------------------------------------------------

def build_pairwise_contrast_prompt(
    archetype_id: str,
    other_id: str,
    self_summary: str,
    other_summary: str,
    self_baseline: str,
    other_baseline: str,
    stain: str,
) -> str:
    return f"""
You are comparing two microscopy archetypes from stain {stain} that were flagged as visually similar.

Image conventions:
- darker grayscale = stronger fluorescence
- red outline = cell boundary
- blue outline = nuclear boundary
- each image is a 5-cell crop

You will be shown two images:
- first image: Archetype {archetype_id}
- second image: Archetype {other_id}

Use only visible image features. Do not infer biology or mechanism.

Reference text:
- Archetype {archetype_id} summary: {self_summary}
- Archetype {other_id} summary: {other_summary}
- Archetype {archetype_id} baseline note: {self_baseline}
- Archetype {other_id} baseline note: {other_baseline}

Task:
- identify only the visual traits that distinguish Archetype {archetype_id} from Archetype {other_id}
- focus on intensity, localization, texture/pattern, cell/nuclear size, cell/nuclear shape, and artifacts
- do not list shared traits

Use exactly this format:

### Distinguishing Features of Archetype {archetype_id} vs Archetype {other_id}:
- ...
- ...
- ...

Be concise, specific, and visual.
""".strip()


def build_refinement_prompt(archetype_id: str, original_summary: str, contrast_notes: str) -> str:
    return f"""
You are producing the final summary for Archetype {archetype_id}.

Your task:
- rewrite the Summary section so it keeps the meaning of the original summary
- fold in the distinguishing visual traits from the contrastive notes, so this archetype reads as distinct from the archetypes it was confused with
- stay observational: no new biology, no invented features
- keep it roughly the same length as the original (~200 words)
- output the summary text only, with no commentary about the task and without repeating the original summary or the contrastive notes

Use exactly this format:

### Summary:
...

---

## Original Summary:
{original_summary.strip()}

---

## Contrastive Notes:
{contrast_notes.strip()}
""".strip()
