"""The five annotation steps, and the filename conventions, report parsing and
summary-similarity helpers they share."""

import base64
import html
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from .llm import (PipelineError, analyze_image, cost_callback_of, image_block, invoke_text,
                  make_llm, verify_image)
from .prompts import (build_baseline_comparison_prompt, build_names_prompt,
                      build_pairwise_contrast_prompt, build_refinement_prompt,
                      build_summary_prompt, get_annotation_prompt)

# ---------------------------------------------------------------------------
# Section headers the pipeline reads and writes. Keep in sync with promptlib.
# ---------------------------------------------------------------------------

SECTION_SUMMARY = "### Summary:"
SECTION_NAMES = "### Proposed Names:"
SECTION_HYPOTHESIS = "### Functional Hypothesis:"
SECTION_REFINEMENT_NOTES = "### Refinement Notes:"
SECTION_CELL_OBSERVATIONS = "### Cell Observations"
SECTION_SHARED_TRAITS = "### Shared Traits"

# Artifact names, so every step agrees on what the previous one wrote.
SYNTHESIS_GLOB = "archetype*_synthesis.md"
FINAL_GLOB = "archetype*_final.md"
ANNOTATIONS_DIR = "archetype_annot"
BASELINE_DIR = "contrastive_baselines"
BASELINE_NOTES = "contrastive_baseline_notes.json"
REVIEWS_DIR = "archetype_reviews"
DASHBOARD = "archetype_dashboard.html"

CALLS = {
    "annotate": dict(temp=0.2, reasoning_effort="low", max_completion_tokens=8000),
    "baseline": dict(temp=0.7, reasoning_effort="low", max_completion_tokens=8000),
    "summary": dict(temp=0.7, reasoning_effort="medium", max_completion_tokens=8000),
    "names": dict(temp=0.2, reasoning_effort="high", max_completion_tokens=8000),
    "refine": dict(temp=0.7, reasoning_effort="high", max_completion_tokens=16000),
}


def synthesis_filename(arch_id: str) -> str:
    return f"archetype{arch_id}_synthesis.md"


def final_filename(arch_id: str) -> str:
    return f"archetype{arch_id}_final.md"


def contrast_filename(arch_id: str, other_id: str) -> str:
    return f"archetype{arch_id}_contrast_vs{other_id}.md"


def check_expected(found_ids, expected_ids, what: str):
    """Fail loudly when the run covers fewer archetypes than the crops step drew."""
    missing = [a for a in sorted_archetype_ids(expected_ids) if a not in found_ids]
    if missing:
        raise PipelineError(f"No {what} for archetype(s) {', '.join(missing)}")


def _arch_sort_key(arch_id: str):
    try:
        return (0, int(arch_id))
    except (TypeError, ValueError):
        return (1, str(arch_id))


def sorted_archetype_ids(ids) -> List[str]:
    return sorted(ids, key=_arch_sort_key)


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

def extract_section(content: str, start_label: str, end_label: str = None) -> str:
    """Pull one '### Label:' block out of a report. Empty string if absent."""
    label = start_label.strip().lstrip("#").strip().rstrip(":")
    if end_label:
        pattern = rf"(?is)#+\s*{re.escape(label)}\s*:?\s*\n(.*?)(?=\n#+\s*(?:{end_label})\s*:?\s*\n|\Z)"
    else:
        pattern = rf"(?is)#+\s*{re.escape(label)}\s*:?\s*\n(.*)"
    match = re.search(pattern, content.replace("\r\n", "\n"))
    return match.group(1).strip() if match else ""


def require_section(content: str, start_label: str, end_label: str, source: str) -> str:
    text = extract_section(content, start_label, end_label)
    if not text:
        raise PipelineError(f"{source}: missing or empty section '{start_label}'")
    return text


def parse_synthesis_report(path) -> Dict[str, str]:
    """Split a step 3 report into its three sections, failing if any is empty."""
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    return {
        "summary": require_section(content, SECTION_SUMMARY, "Proposed Names|Functional Hypothesis", path.name),
        "names": require_section(content, SECTION_NAMES, "Functional Hypothesis", path.name),
        "hypothesis": require_section(content, SECTION_HYPOTHESIS, None, path.name),
    }


def load_contrastive_baselines(path) -> Dict[str, str]:
    path = Path(path)
    if not path.exists():
        raise PipelineError(f"Contrastive baseline JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    notes = data.get("per_archetype_notes", {})
    if not notes:
        raise PipelineError(f"{path} contains no per_archetype_notes")
    return notes


# ---------------------------------------------------------------------------
# Summary similarity (step 4 pair selection)
# ---------------------------------------------------------------------------

_hf_model = None


def _embedder():
    global _hf_model
    if _hf_model is None:
        from sentence_transformers import SentenceTransformer
        _hf_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _hf_model


def compute_pairwise_similarities(
    summary_data: Dict[str, str],
    threshold: float = 0.1,
    max_pairs_per_archetype: int = 0,
) -> List[Tuple[str, str]]:
    """Archetype pairs whose summaries sit within `threshold` cosine distance.

    Summaries from one stain are near-duplicate prose, so this fires often and
    each pair costs two LLM calls in step 4. max_pairs_per_archetype > 0 keeps
    only the closest N partners per archetype and is the lever for capping that
    cost; 0 keeps every pair under the threshold.
    """
    from scipy.spatial.distance import cosine

    if not summary_data:
        raise PipelineError("No summaries provided for similarity computation")

    arch_ids = sorted_archetype_ids(summary_data.keys())
    embeddings = {aid: _embedder().encode(summary_data[aid], convert_to_numpy=True) for aid in arch_ids}

    scored = []
    for i, a in enumerate(arch_ids):
        for b in arch_ids[i + 1:]:
            dist = float(cosine(embeddings[a], embeddings[b]))
            logging.info("archetype %s vs %s -> cosine distance %.3f", a, b, dist)
            if dist < threshold:
                scored.append((dist, a, b))

    scored.sort()
    if max_pairs_per_archetype > 0:
        kept, counts = [], {}
        for dist, a, b in scored:
            if counts.get(a, 0) >= max_pairs_per_archetype or counts.get(b, 0) >= max_pairs_per_archetype:
                logging.info("capping: dropping pair %s/%s (distance %.3f)", a, b, dist)
                continue
            counts[a] = counts.get(a, 0) + 1
            counts[b] = counts.get(b, 0) + 1
            kept.append((dist, a, b))
        dropped = len(scored) - len(kept)
        if dropped:
            logging.info(f"Pair cap {max_pairs_per_archetype}/archetype dropped {dropped} of {len(scored)} pairs")
        scored = kept

    return [(a, b) for _, a, b in scored]


# ---------------------------------------------------------------------------
# Step 1 - per-image annotation
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = (SECTION_CELL_OBSERVATIONS, SECTION_SHARED_TRAITS)


def annotation_path(output_folder: Path, arch_id: str, grid_id: int) -> Path:
    return output_folder / f"archetype{arch_id}_grid_{grid_id}_annotation.md"


def write_report(path: Path, arch_id: str, grid_id: int, image_path: Path, model: str, body: str):
    header = "\n".join([
        f"# Archetype {arch_id} - grid {grid_id}",
        f"- Source image: {image_path.name}",
        f"- Model: {model}",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ])
    path.write_text(header + body.strip() + "\n", encoding="utf-8")


def annotate(model: str, stain: str, grids, directory: Path, resume: bool = False):
    """One vision call per grid image, producing a purely observational report."""
    output_folder = directory / ANNOTATIONS_DIR
    output_folder.mkdir(parents=True, exist_ok=True)

    arch_ids = sorted_archetype_ids(grids)
    logging.info(f"{stain}: {len(arch_ids)} archetypes -> {', '.join(arch_ids)}")

    llm = make_llm(model, **CALLS["annotate"])

    written, reused = 0, 0
    for arch_id in arch_ids:
        for grid_id, image_path in enumerate(grids[arch_id], start=1):
            out_path = annotation_path(output_folder, arch_id, grid_id)
            if resume and out_path.exists() and out_path.stat().st_size > 0:
                logging.info(f"archetype {arch_id} grid {grid_id}: reusing {out_path.name}")
                reused += 1
                continue

            width, height = verify_image(image_path)
            logging.info(f"archetype {arch_id} grid {grid_id}: {image_path.name} ({width}x{height})")
            body = analyze_image(
                llm,
                image_path,
                get_annotation_prompt(stain),
                context=f"step1 {stain} archetype {arch_id} grid {grid_id}",
                require=REQUIRED_SECTIONS,
                detail="high",
            )
            write_report(out_path, arch_id, grid_id, image_path, model, body)
            logging.info(f"{out_path.name} ({len(body)} chars)")
            written += 1

    # Every archetype must end up with a grid_1 annotation; step 3 depends on it.
    missing = [
        arch_id for arch_id in arch_ids
        if not annotation_path(output_folder, arch_id, 1).exists()
    ]
    if missing:
        raise PipelineError(
            f"No grid_1 annotation produced for archetype(s): {', '.join(missing)}"
        )

    cost = cost_callback_of(llm)
    logging.info(f"Step 1 complete: {written} written, {reused} reused, {len(arch_ids)} archetypes covered")
    if cost:
        logging.info(cost.report())


# ---------------------------------------------------------------------------
# Step 2 - contrastive baseline
# ---------------------------------------------------------------------------

MAX_EDGE = 1024


def compare_to_baseline(llm, arch_id, stain, normal_ids, image_paths, detail):
    prompt = build_baseline_comparison_prompt(arch_id, stain, normal_ids)
    content = [{"type": "text", "text": prompt}]
    content.extend(image_block(p, detail=detail, max_edge=MAX_EDGE) for p in image_paths)
    return invoke_text(
        llm,
        [{"role": "user", "content": content}],
        context=f"step2 {stain} archetype {arch_id} vs baseline",
        require=(f"Archetype {arch_id}",),
    )


def baseline(model: str, stain: str, grids, normal_ids, directory: Path):
    """Each archetype's grid_1 image against the normal archetypes', in one request."""
    out_dir = directory / BASELINE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    images_by_arch = {arch_id: paths[0] for arch_id, paths in grids.items()}
    logging.info(f"Found {len(images_by_arch)} archetypes with grid_1 images")

    normal_ids = list(normal_ids or [])
    if not normal_ids:
        raise PipelineError("At least one normal archetype is required for the baseline")

    missing = [a for a in normal_ids if a not in images_by_arch]
    if missing:
        raise PipelineError(
            f"Missing grid_1 image(s) for normal archetype(s): {', '.join(missing)}"
        )
    logging.info(f"Baseline archetype(s): {', '.join(normal_ids)}")

    llm = make_llm(model, **CALLS["baseline"])

    normal_paths = [images_by_arch[a] for a in normal_ids]
    per_arch = {}
    for arch_id in sorted_archetype_ids(images_by_arch):
        logging.info(f"Comparing archetype {arch_id} to baseline...")
        note = compare_to_baseline(
            llm,
            arch_id,
            stain,
            normal_ids,
            [images_by_arch[arch_id]] + normal_paths,
            "low",
        )
        per_arch[arch_id] = note
        logging.info(f"{len(note)} chars")

    if len(per_arch) != len(images_by_arch):
        raise PipelineError(
            f"Produced {len(per_arch)} baseline notes for {len(images_by_arch)} archetypes"
        )

    output_json = out_dir / BASELINE_NOTES
    output_json.write_text(
        json.dumps({"stain": stain, "normal_ids": normal_ids,
                    "per_archetype_notes": per_arch}, indent=2),
        encoding="utf-8",
    )

    report = [
        f"# Contrastive Baseline Report - {stain.upper()}",
        "",
        f"Baseline archetype(s): {', '.join(normal_ids)}",
        "",
        "## Per-Archetype Notes",
        "",
    ]
    for arch_id in sorted_archetype_ids(per_arch):
        report.append(per_arch[arch_id])
        report.append("")
    (out_dir / "contrastive_baseline_report.md").write_text(
        "\n".join(report).strip() + "\n", encoding="utf-8"
    )

    # Read the JSON back so a malformed write fails here, not in step 3.
    reloaded = json.loads(output_json.read_text(encoding="utf-8"))["per_archetype_notes"]
    if set(reloaded) != set(per_arch):
        raise PipelineError(f"{output_json} did not round-trip correctly")

    cost = cost_callback_of(llm)
    logging.info(f"Step 2 complete: {len(per_arch)} baseline notes -> {output_json}")
    if cost:
        logging.info(cost.report())


# ---------------------------------------------------------------------------
# Step 3 - synthesis, names and hypothesis
# ---------------------------------------------------------------------------

def load_annotations(report_folder: Path, grid_id: int = 1):
    """archetype_id -> grid_1 annotation text."""
    reports = {}
    for path in sorted(report_folder.glob(f"archetype*_grid_{grid_id}_annotation.md")):
        arch_id = path.name.split("_")[0].replace("archetype", "")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise PipelineError(f"Empty step 1 annotation: {path}")
        reports[arch_id] = text
    if not reports:
        raise PipelineError(
            f"No archetype*_grid_{grid_id}_annotation.md files in {report_folder}"
        )
    return reports


def synthesize(model: str, stain: str, grids, directory: Path):
    """Text only: the grid_1 annotation plus the baseline note become a summary,
    then a second call proposes names and one functional hypothesis."""
    output_dir = directory / REVIEWS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_annotations(directory / ANNOTATIONS_DIR)
    baselines = load_contrastive_baselines(directory / BASELINE_DIR / BASELINE_NOTES)
    check_expected(annotations, grids, f"{stain} annotations")
    annotations = {a: annotations[a] for a in grids}

    missing_notes = [a for a in annotations if a not in baselines]
    if missing_notes:
        raise PipelineError(
            "No baseline note for archetype(s): " + ", ".join(sorted_archetype_ids(missing_notes))
        )

    llm_summary = make_llm(model, **CALLS["summary"])
    llm_names = make_llm(model, **CALLS["names"])

    arch_ids = sorted_archetype_ids(annotations)
    logging.info(f"Synthesizing {len(arch_ids)} archetypes: {', '.join(arch_ids)}")

    combined = []
    for arch_id in arch_ids:
        annotation = annotations[arch_id]
        baseline_note = baselines.get(arch_id, "")

        logging.info(f"Archetype {arch_id}: summary...")
        summary_block = invoke_text(
            llm_summary,
            [{"role": "user", "content": build_summary_prompt(arch_id, annotation, baseline_note)}],
            context=f"step3 {stain} archetype {arch_id} summary",
            require=(SECTION_SUMMARY,),
        )

        logging.info(f"Archetype {arch_id}: names and hypothesis...")
        names_block = invoke_text(
            llm_names,
            [{"role": "user", "content": build_names_prompt(arch_id, annotation, stain)}],
            context=f"step3 {stain} archetype {arch_id} names",
            require=(SECTION_NAMES, SECTION_HYPOTHESIS),
        )

        body = f"{summary_block.strip()}\n\n{names_block.strip()}\n"
        out_path = output_dir / synthesis_filename(arch_id)
        out_path.write_text(body, encoding="utf-8")

        # Fail here rather than let step 4 inherit an unparseable report.
        parse_synthesis_report(out_path)
        logging.info(f"{out_path.name}")
        combined.append(f"## Archetype {arch_id}\n\n{body}\n---\n")

    header = (
        f"# Archetype Synthesis - {stain.upper()}\n"
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with {model}_\n\n"
    )
    (output_dir / "combined_archetype_synthesis.md").write_text(
        header + "\n".join(combined), encoding="utf-8"
    )

    logging.info(f"Step 3 complete: {len(arch_ids)} synthesis reports in {output_dir}")
    for llm, label in ((llm_summary, "summary"), (llm_names, "names")):
        cost = cost_callback_of(llm)
        if cost:
            logging.info(f"{label}: {cost.report()}")


# ---------------------------------------------------------------------------
# Step 4 - contrastive refinement, and the final report
# ---------------------------------------------------------------------------

MIN_SUMMARY_CHARS = 150


def run_pairwise_contrasts(llm, pairs, summaries, baselines, grids, stain,
                           review_dir, image_detail):
    """Both directions of every flagged pair. Returns arch_id -> [notes]."""
    notes = {}
    total = len(pairs) * 2
    done = 0
    for arch_a, arch_b in pairs:
        block_a = image_block(grids[arch_a][0], detail=image_detail, max_edge=1024)
        block_b = image_block(grids[arch_b][0], detail=image_detail, max_edge=1024)

        for this_id, other_id, this_block, other_block in (
            (arch_a, arch_b, block_a, block_b),
            (arch_b, arch_a, block_b, block_a),
        ):
            done += 1
            logging.info(f"[{done}/{total}] contrasting archetype {this_id} against {other_id}...")
            prompt = build_pairwise_contrast_prompt(
                archetype_id=this_id,
                other_id=other_id,
                self_summary=summaries.get(this_id, ""),
                other_summary=summaries.get(other_id, ""),
                self_baseline=baselines.get(this_id, "*No baseline note*"),
                other_baseline=baselines.get(other_id, "*No baseline note*"),
                stain=stain,
            )
            text = invoke_text(
                llm,
                [{"role": "user", "content": [
                    {"type": "text", "text": prompt}, this_block, other_block,
                ]}],
                context=f"step4 {stain} archetype {this_id} vs {other_id}",
                require=(f"Archetype {this_id}",),
            )
            path = Path(review_dir) / contrast_filename(this_id, other_id)
            path.write_text(text.strip() + "\n", encoding="utf-8")
            notes.setdefault(this_id, []).append(text.strip())
            logging.info(f"{path.name} ({len(text)} chars)")
    return notes


def refine_summary(llm, arch_id, original_summary, contrast_notes, stain):
    """Rewrite one summary against its contrastive notes."""
    reply = invoke_text(
        llm,
        [{"role": "user", "content": build_refinement_prompt(
            arch_id, original_summary, "\n\n".join(contrast_notes))}],
        context=f"step4 {stain} archetype {arch_id} refinement",
        require=(SECTION_SUMMARY,),
    )
    # The model sometimes echoes the Original Summary / Contrastive Notes blocks
    # back; keep only the Summary section.
    refined = extract_section(reply, SECTION_SUMMARY, "Original Summary|Contrastive Notes")
    if len(refined) < MIN_SUMMARY_CHARS:
        raise PipelineError(
            f"step4 {stain} archetype {arch_id}: refined summary is too short "
            f"({len(refined)} chars, minimum {MIN_SUMMARY_CHARS}). "
            f"Reply began: {reply[:200]!r}"
        )
    return refined


def write_final_report(path: Path, arch_id, stain, model, summary, names, hypothesis,
                       compared_with, refined: bool):
    if refined:
        note = ("- Summary refined against archetype(s): "
                + ", ".join(sorted_archetype_ids(compared_with)))
    else:
        note = "- No archetype was similar enough to require contrastive refinement; summary carried over from synthesis."

    path.write_text("\n".join([
        f"{SECTION_SUMMARY}",
        summary.strip(),
        "",
        f"{SECTION_NAMES}",
        names.strip(),
        "",
        f"{SECTION_HYPOTHESIS}",
        hypothesis.strip(),
        "",
        f"{SECTION_REFINEMENT_NOTES}",
        note,
        f"- Stain: {stain} | Archetype: {arch_id} | Model: {model}",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]), encoding="utf-8")


def refine(model: str, stain: str, grids, directory: Path, similarity_threshold: float,
           max_pairs_per_archetype: int, iterations: int = 1):
    """Pairs of summaries that read too much alike get a two-image contrastive
    call in both directions, and each summary is rewritten to fold in what
    makes it distinct.  Writes archetype{i}_final.md."""
    review_dir = directory / REVIEWS_DIR
    baselines = load_contrastive_baselines(directory / BASELINE_DIR / BASELINE_NOTES)

    # Sections that survive refinement unchanged.
    sections = {}
    for path in sorted(review_dir.glob(SYNTHESIS_GLOB)):
        arch_id = path.name.split("_")[0].replace("archetype", "")
        if arch_id in grids:
            sections[arch_id] = parse_synthesis_report(path)
    check_expected(sections, grids, f"{stain} synthesis reports")

    arch_ids = sorted_archetype_ids(sections)
    summaries = {a: sections[a]["summary"] for a in arch_ids}
    logging.info(f"Loaded {len(arch_ids)} synthesis reports: {', '.join(arch_ids)}")

    llm = make_llm(model, **CALLS["refine"])

    compared_with = {a: set() for a in arch_ids}
    refined_flag = {a: False for a in arch_ids}

    for iteration in range(1, iterations + 1):
        logging.info(f"Iteration {iteration}/{iterations}")
        pairs = compute_pairwise_similarities(
            summaries,
            threshold=similarity_threshold,
            max_pairs_per_archetype=max_pairs_per_archetype,
        )
        if not pairs:
            logging.info("No archetype pairs below the similarity threshold this round.")
            break

        logging.info(f"{len(pairs)} similar pair(s) -> {2 * len(pairs)} contrastive calls")
        notes = run_pairwise_contrasts(
            llm, pairs, summaries, baselines, grids, stain, review_dir, "low",
        )

        for arch_a, arch_b in pairs:
            compared_with[arch_a].add(arch_b)
            compared_with[arch_b].add(arch_a)

        for arch_id in sorted_archetype_ids(notes):
            logging.info(f"Refining summary for archetype {arch_id}...")
            summaries[arch_id] = refine_summary(
                llm, arch_id, summaries[arch_id], notes[arch_id], stain
            )
            refined_flag[arch_id] = True
            logging.info(f"{len(summaries[arch_id])} chars")

    combined = []
    for arch_id in arch_ids:
        out_path = review_dir / final_filename(arch_id)
        write_final_report(
            out_path,
            arch_id,
            stain,
            model,
            summaries[arch_id],
            sections[arch_id]["names"],
            sections[arch_id]["hypothesis"],
            compared_with[arch_id],
            refined_flag[arch_id],
        )
        content = out_path.read_text(encoding="utf-8")
        for marker in (SECTION_SUMMARY, SECTION_NAMES, SECTION_HYPOTHESIS):
            if marker not in content:
                raise PipelineError(f"{out_path.name} is missing {marker}")
        combined.append(f"## Archetype {arch_id}\n\n{content}\n---\n")
        logging.info(f"{out_path.name} ({'refined' if refined_flag[arch_id] else 'carried over'})")

    header = (
        f"# Final Archetype Reports - {stain.upper()}\n"
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with {model}_\n\n"
    )
    (review_dir / "combined_archetype_final.md").write_text(
        header + "\n".join(combined), encoding="utf-8"
    )

    n_refined = sum(refined_flag.values())
    logging.info(f"Step 4 complete: {len(arch_ids)} final reports "
                 f"({n_refined} refined, {len(arch_ids) - n_refined} carried over)")
    cost = cost_callback_of(llm)
    if cost:
        logging.info(cost.report())


# ---------------------------------------------------------------------------
# Step 5 - dashboard
# ---------------------------------------------------------------------------

STYLE = """
body { font-family: sans-serif; max-width: 1000px; margin: auto; padding: 2em; line-height: 1.5; color: #222; }
h1 { margin-bottom: 0.35em; }
h2 { border-bottom: 1px solid #ccc; padding-bottom: 0.3em; margin-bottom: 0.4em; }
.section { margin-bottom: 4em; }
.meta { color: #555; margin-bottom: 1em; }
.images { margin: 1em 0 1.5em; }
.image-block { margin-bottom: 1.5em; }
img.full-width { width: 100%; margin: 0.4em 0; border: 1px solid #ccc; }
.label { font-weight: bold; margin-top: 1em; display: block; font-size: 1.05em; }
.report { white-space: pre-wrap; word-wrap: break-word; background: #f7f7f7; border: 1px solid #ddd; padding: 1em; }
""".strip()


def embed_image(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "jpeg" if suffix in {".jpg", ".jpeg"} else "gif" if suffix == ".gif" else "png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return (f'<img src="data:image/{mime};base64,{encoded}" '
            f'class="full-width" alt="{html.escape(path.name)}"/>')


def build_dashboard(review_dir: Path, stain: str, grids):
    found = {re.search(r"archetype(\d+)", p.name).group(1): p for p in review_dir.glob(FINAL_GLOB)}
    check_expected(found, grids, f"{stain} final reports")
    reports = [found[a] for a in sorted_archetype_ids(grids)]

    doc = [
        "<html><head><meta charset='UTF-8'>",
        f"<title>Archetype Dashboard - {html.escape(stain)}</title>",
        f"<style>{STYLE}</style></head><body>",
        f"<h1>🧬 Archetype Dashboard - Stain: {html.escape(stain)}</h1><hr>",
        f"<div class='meta'>Stain: {html.escape(stain)} | Reports: {len(reports)}</div>",
    ]

    for path in reports:
        arch_id = re.search(r"archetype(\d+)", path.name).group(1)
        report_text = path.read_text(encoding="utf-8").strip()
        if not report_text:
            raise PipelineError(f"Empty final report: {path}")
        if not extract_section(report_text, SECTION_SUMMARY, "Proposed Names|Functional Hypothesis"):
            raise PipelineError(f"{path.name} has no '{SECTION_SUMMARY}' section")

        doc.append("<div class='section'>")
        doc.append(f"<h2>Archetype {html.escape(arch_id)}</h2>")
        doc.append(f"<div class='meta'>Source report: {html.escape(path.name)}</div>")
        doc.append("<div class='label'>Images:</div>")
        doc.append("<div class='images'>")
        for grid_idx, img_path in enumerate(grids[arch_id], start=1):
            doc.append("<div class='image-block'>")
            doc.append(f"<div class='meta'>grid_{grid_idx}</div>")
            doc.append(embed_image(img_path))
            doc.append("</div>")
        doc.append("</div>")
        doc.append("<div class='label'>Final Archetype Report:</div>")
        doc.append(f"<div class='report'>{html.escape(report_text)}</div>")
        doc.append("</div>")

    doc.append("</body></html>")
    return "\n".join(doc)


def dashboard(stain: str, grids, directory: Path) -> Path:
    """The final reports next to their grid images as one standalone HTML file,
    with the images embedded so it can be emailed or opened from anywhere."""
    output_path = directory / DASHBOARD
    output_path.write_text(build_dashboard(directory / REVIEWS_DIR, stain, grids), encoding="utf-8")
    size_mb = output_path.stat().st_size / 1e6
    logging.info(f"Dashboard written: {output_path} ({len(grids)} archetypes, {size_mb:.1f} MB)")
    return output_path
