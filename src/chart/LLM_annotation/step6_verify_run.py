#!/usr/bin/env python3
"""Step 6 - verify the run.

Walks every artifact a stain should have produced and exits non-zero if
anything is missing, empty, or malformed. This is the backstop for "no step
silently fails": even if a step wrote a degenerate file, the run does not get
to report success. All problems are listed at once, not just the first.

Checks per archetype:
    step 1  archetype{i}_grid_1_annotation.md  exists, non-trivial, has both sections
    step 2  a baseline note in the JSON
    step 3  archetype{i}_synthesis.md          parses into all three sections
    step 4  archetype{i}_final.md              parses, Summary is substantial
    step 4  every contrast note file non-empty
    step 5  dashboard exists, mentions every archetype
"""

import argparse
import json
import re
import sys
from pathlib import Path

from utils import (
    SECTION_CELL_OBSERVATIONS,
    SECTION_HYPOTHESIS,
    SECTION_NAMES,
    SECTION_SHARED_TRAITS,
    SECTION_SUMMARY,
    extract_section,
    sorted_archetype_ids,
)

MIN_ANNOTATION_CHARS = 400
MIN_SUMMARY_CHARS = 150


def check_file(path: Path, min_chars: int, problems: list, label: str):
    """Record size problems but still return the text, so the structural checks
    downstream also run and the report names every fault at once."""
    if not path.exists():
        problems.append(f"{label}: missing {path}")
        return None
    text = path.read_text(encoding="utf-8").strip()
    if len(text) < min_chars:
        problems.append(f"{label}: {path.name} is only {len(text)} chars (minimum {min_chars})")
    return text


def main(args) -> int:
    annot_dir = Path(args.annotation_folder)
    review_dir = Path(args.review_folder)
    problems = []

    # Which archetypes should exist: from the step 2 JSON, which covers them all.
    baseline_path = Path(args.contrastive_json)
    if not baseline_path.exists():
        print(f"❌ Verification failed: step 2 output missing ({baseline_path})", file=sys.stderr)
        return 1
    baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
    notes = baseline_data.get("per_archetype_notes", {})
    arch_ids = sorted_archetype_ids(notes)
    if not arch_ids:
        print(f"❌ Verification failed: {baseline_path} has no per_archetype_notes", file=sys.stderr)
        return 1

    if args.expected_archetypes and len(arch_ids) != args.expected_archetypes:
        problems.append(
            f"step2: {len(arch_ids)} archetypes in the baseline JSON, "
            f"expected {args.expected_archetypes}"
        )

    for arch_id in arch_ids:
        # Step 1
        text = check_file(
            annot_dir / f"archetype{arch_id}_grid_1_annotation.md",
            MIN_ANNOTATION_CHARS, problems, f"step1 A{arch_id}",
        )
        if text:
            for marker in (SECTION_CELL_OBSERVATIONS, SECTION_SHARED_TRAITS):
                if marker.lower() not in text.lower():
                    problems.append(f"step1 A{arch_id}: annotation has no '{marker}'")

        # Step 2
        if not (notes.get(arch_id) or "").strip():
            problems.append(f"step2 A{arch_id}: empty baseline note")

        # Step 3
        text = check_file(
            review_dir / f"archetype{arch_id}_synthesis.md",
            MIN_SUMMARY_CHARS, problems, f"step3 A{arch_id}",
        )
        if text:
            for marker, end in (
                (SECTION_SUMMARY, "Proposed Names|Functional Hypothesis"),
                (SECTION_NAMES, "Functional Hypothesis"),
                (SECTION_HYPOTHESIS, None),
            ):
                if not extract_section(text, marker, end):
                    problems.append(f"step3 A{arch_id}: synthesis missing '{marker}'")

        # Step 4
        text = check_file(
            review_dir / f"archetype{arch_id}_final.md",
            MIN_SUMMARY_CHARS, problems, f"step4 A{arch_id}",
        )
        if text:
            summary = extract_section(text, SECTION_SUMMARY, "Proposed Names|Functional Hypothesis")
            if len(summary) < MIN_SUMMARY_CHARS:
                problems.append(
                    f"step4 A{arch_id}: final Summary is {len(summary)} chars "
                    f"(minimum {MIN_SUMMARY_CHARS})"
                )
            for marker, end in ((SECTION_NAMES, "Functional Hypothesis"),
                                (SECTION_HYPOTHESIS, "Refinement Notes")):
                if not extract_section(text, marker, end):
                    problems.append(f"step4 A{arch_id}: final report missing '{marker}'")

    # Step 4 contrast notes
    contrast_files = list(review_dir.glob("archetype*_contrast_vs*.md"))
    empty_contrasts = [p.name for p in contrast_files if p.stat().st_size == 0]
    if empty_contrasts:
        problems.append(
            f"step4: {len(empty_contrasts)} empty contrast note(s): "
            + ", ".join(empty_contrasts[:5])
            + ("..." if len(empty_contrasts) > 5 else "")
        )

    # Step 5
    if args.dashboard:
        dash = Path(args.dashboard)
        if not dash.exists():
            problems.append(f"step5: dashboard missing ({dash})")
        else:
            dash_text = dash.read_text(encoding="utf-8")
            rendered = set(re.findall(r"<h2>Archetype\s+(\d+)</h2>", dash_text))
            absent = [a for a in arch_ids if a not in rendered]
            if absent:
                problems.append(f"step5: dashboard has no section for archetype(s) {', '.join(absent)}")

    print(f"🔎 Verified {len(arch_ids)} archetypes for {args.stain}: "
          f"{len(contrast_files)} contrast note(s)")
    if problems:
        print(f"\n❌ Verification failed with {len(problems)} problem(s):", file=sys.stderr)
        for p in problems:
            print(f"   - {p}", file=sys.stderr)
        return 1

    print(f"✅ {args.stain}: all artifacts present and well-formed")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Step 6: verify all pipeline artifacts")
    parser.add_argument("--stain", required=True)
    parser.add_argument("--annotation_folder", required=True)
    parser.add_argument("--review_folder", required=True)
    parser.add_argument("--contrastive_json", required=True)
    parser.add_argument("--dashboard", default=None)
    parser.add_argument("--expected_archetypes", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main(parse_args()))
