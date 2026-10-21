# utils.py
"""Filename conventions, report parsing, and summary-similarity helpers."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

from llm_core import PipelineError

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


def synthesis_filename(arch_id: str) -> str:
    return f"archetype{arch_id}_synthesis.md"


def final_filename(arch_id: str) -> str:
    return f"archetype{arch_id}_final.md"


def contrast_filename(arch_id: str, other_id: str) -> str:
    return f"archetype{arch_id}_contrast_vs{other_id}.md"


# ---------------------------------------------------------------------------
# Stain file tags
# ---------------------------------------------------------------------------

STAIN_FILE_TAG_ALIASES = {
    "TUBULIN": ["B-TUBULIN"],
    "YH2AX": ["yH2AX"],
}


def stain_file_tags(stain: str) -> List[str]:
    tags = [stain, stain.upper(), stain.lower()]
    tags.extend(STAIN_FILE_TAG_ALIASES.get(stain.upper(), []))
    return list(dict.fromkeys(tags))


def grid_image_path(image_folder, arch_id: str, stain: str, grid_id: int):
    """The grid image for one archetype, or None if no tag spelling matches."""
    folder = Path(image_folder)
    for tag in stain_file_tags(stain):
        candidate = folder / f"archetype{arch_id}_{tag}_grid_{grid_id}.png"
        if candidate.exists():
            return candidate
    return None


def require_grid_image(image_folder, arch_id: str, stain: str, grid_id: int) -> Path:
    path = grid_image_path(image_folder, arch_id, stain, grid_id)
    if path is None:
        tried = ", ".join(
            f"archetype{arch_id}_{tag}_grid_{grid_id}.png" for tag in stain_file_tags(stain)
        )
        raise PipelineError(
            f"No grid_{grid_id} image for archetype {arch_id} in {image_folder}. Tried: {tried}"
        )
    return path


def discover_archetypes(image_folder, stain: str, grid_id: int = 1) -> Dict[str, Path]:
    """archetype_id -> grid image path, for every archetype present."""
    found = {}
    tags = {tag.lower() for tag in stain_file_tags(stain)}
    for path in sorted(Path(image_folder).glob(f"archetype*_grid_{grid_id}.png")):
        if any(tag in path.name.lower() for tag in tags):
            arch_id = path.name.split("_")[0].replace("archetype", "")
            found[arch_id] = path
    return found


# Prefixing a grid image takes that archetype out of the run while leaving the
# file on disk: `exclude_archetype8_KI67_grid_1.png`, `no-cyto_archetype3_…`.
# Those archetypes are accounted for, not missing.
_EXCLUDED_RE = re.compile(r"^(?P<prefix>.+?)_archetype(?P<id>\d+)_", re.IGNORECASE)


def discover_excluded_archetypes(image_folder, stain: str, grid_id: int = 1) -> Dict[str, str]:
    """archetype_id -> the prefix that excluded it."""
    excluded = {}
    tags = {tag.lower() for tag in stain_file_tags(stain)}
    for path in sorted(Path(image_folder).glob(f"*archetype*_grid_{grid_id}.png")):
        if path.name.startswith("archetype"):
            continue
        if not any(tag in path.name.lower() for tag in tags):
            continue
        match = _EXCLUDED_RE.match(path.name)
        if match:
            excluded[match.group("id")] = match.group("prefix")
    return excluded


def describe_exclusions(excluded: Dict[str, str]) -> str:
    return ", ".join(
        f"A{arch_id} ({prefix}_)" for arch_id, prefix in
        sorted(excluded.items(), key=lambda kv: _arch_sort_key(kv[0]))
    )


def check_expected_count(found_ids, expected: int, what: str, excluded: Dict[str, str] = None):
    """Fail loudly when the run covers fewer archetypes than configured.

    `expected` is the clustering k. Archetypes deliberately excluded by renaming
    their image count toward it, so k stays the number the clustering produced;
    anything unaccounted for is a real gap and raises.
    """
    if not expected:
        return
    excluded = excluded or {}
    overlap = set(found_ids) & set(excluded)
    if overlap:
        raise PipelineError(
            f"{what}: archetype(s) {', '.join(sorted(overlap, key=_arch_sort_key))} have both "
            "an active and an excluded grid image; remove one of the two files"
        )
    total = len(found_ids) + len(excluded)
    if total != expected:
        raise PipelineError(
            f"Expected {expected} {what} but accounted for {total} "
            f"({len(found_ids)} found, {len(excluded)} excluded). "
            f"Found: {', '.join(sorted(found_ids, key=_arch_sort_key)) or 'none'}. "
            f"Excluded: {describe_exclusions(excluded) or 'none'}. "
            "Either an image is missing, or k in STAIN_CONFIG does not match the clustering."
        )


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
            print(f"ℹ️  Pair cap {max_pairs_per_archetype}/archetype dropped {dropped} of {len(scored)} pairs")
        scored = kept

    return [(a, b) for _, a, b in scored]
