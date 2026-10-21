#!/usr/bin/env python3
"""Step 2 - contrastive baseline.

For each archetype, send its grid_1 image together with the designated normal
archetype grid(s) in one request and record how it differs from that baseline.
The note produced here is what lets step 3 make relative statements such as
"nuclei larger than baseline".

Outputs:
    <output_folder>/contrastive_baseline_notes.json   (consumed by steps 3 and 4)
    <output_folder>/contrastive_baseline_report.md    (human-readable)
"""

import argparse
import json
import sys
from pathlib import Path

from image_utils import image_block
from llm_core import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_MODEL,
    PipelineError,
    check_llm_credentials,
    cost_callback_of,
    invoke_text,
    make_llm,
)
from promptlib import build_baseline_comparison_prompt
from utils import (
    check_expected_count,
    describe_exclusions,
    discover_archetypes,
    discover_excluded_archetypes,
    sorted_archetype_ids,
)

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


def main(args) -> int:
    image_folder = Path(args.image_folder)
    if not image_folder.is_dir():
        raise PipelineError(f"Image folder not found: {image_folder}")

    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔌 Provider: {check_llm_credentials(args.provider, args.model)}")

    images_by_arch = discover_archetypes(image_folder, args.stain, grid_id=1)
    if not images_by_arch:
        raise PipelineError(f"No grid_1 images for stain '{args.stain}' in {image_folder}")
    excluded = discover_excluded_archetypes(image_folder, args.stain, grid_id=1)
    check_expected_count(images_by_arch, args.expected_archetypes, f"{args.stain} archetypes", excluded)
    print(f"🧬 Found {len(images_by_arch)} archetypes with grid_1 images")
    if excluded:
        print(f"🚫 excluded by filename prefix: {describe_exclusions(excluded)}")

    normal_ids = list(args.normal or [])
    if not normal_ids:
        raise PipelineError("At least one --normal archetype is required for the baseline")

    missing = [a for a in normal_ids if a not in images_by_arch]
    if missing:
        raise PipelineError(
            f"Missing grid_1 image(s) for normal archetype(s): {', '.join(missing)}"
        )
    print(f"📐 Baseline archetype(s): {', '.join(normal_ids)}")

    llm = make_llm(
        args.model,
        provider=args.provider,
        temp=args.temperature,
        reasoning_effort=args.reasoning_effort,
        verbosity=args.verbosity,
        max_completion_tokens=args.max_completion_tokens,
    )

    normal_paths = [images_by_arch[a] for a in normal_ids]
    per_arch = {}
    for arch_id in sorted_archetype_ids(images_by_arch):
        print(f"📡 Comparing archetype {arch_id} to baseline...")
        note = compare_to_baseline(
            llm,
            arch_id,
            args.stain,
            normal_ids,
            [images_by_arch[arch_id]] + normal_paths,
            args.image_detail,
        )
        per_arch[arch_id] = note
        print(f"   ✅ {len(note)} chars")

    if len(per_arch) != len(images_by_arch):
        raise PipelineError(
            f"Produced {len(per_arch)} baseline notes for {len(images_by_arch)} archetypes"
        )

    output_json = out_dir / "contrastive_baseline_notes.json"
    output_json.write_text(
        json.dumps({"stain": args.stain, "normal_ids": normal_ids,
                    "per_archetype_notes": per_arch}, indent=2),
        encoding="utf-8",
    )

    report = [
        f"# Contrastive Baseline Report - {args.stain.upper()}",
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
    print(f"\n✅ Step 2 complete: {len(per_arch)} baseline notes -> {output_json}")
    if cost:
        print(f"💸 {cost.report()}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Step 2: contrastive baseline comparison")
    parser.add_argument("--stain", required=True)
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--normal", nargs="+", required=True,
                        help="Archetype id(s) that define the normal baseline")
    parser.add_argument("--expected_archetypes", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider", default=None,
                        help="gateway | azure | auto (default: $LLM_PROVIDER, else auto)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--reasoning_effort", default="low")
    parser.add_argument("--verbosity", default=None)
    parser.add_argument("--max_completion_tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    parser.add_argument("--image_detail", default="low", choices=["low", "high", "auto"])
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.exit(main(parse_args()))
    except PipelineError as exc:
        print(f"\n❌ Step 2 failed: {exc}", file=sys.stderr)
        sys.exit(1)
