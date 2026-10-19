#!/usr/bin/env python3
"""Step 1 - per-image annotation.

One vision call per grid image, producing a purely observational report.
Runs a whole stain in a single process so the archetype count can be checked
against the configured k instead of quietly annotating whatever happens to be
on disk.

Output: <output_folder>/archetype{i}_grid_{n}_annotation.md
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from image_utils import analyze_image, verify_image
from llm_core import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_MODEL,
    PipelineError,
    check_llm_credentials,
    cost_callback_of,
    make_llm,
    model_supports_temperature,
)
from promptlib import get_annotation_prompt
from utils import (
    SECTION_CELL_OBSERVATIONS,
    SECTION_SHARED_TRAITS,
    check_expected_count,
    describe_exclusions,
    discover_archetypes,
    discover_excluded_archetypes,
    grid_image_path,
    require_grid_image,
    sorted_archetype_ids,
)

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


def main(args) -> int:
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    image_folder = Path(args.image_folder)
    if not image_folder.is_dir():
        raise PipelineError(f"Image folder not found: {image_folder}")

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"🔌 Provider: {check_llm_credentials(args.provider, args.model)}")

    archetypes = discover_archetypes(image_folder, args.stain, grid_id=1)
    if not archetypes:
        raise PipelineError(
            f"No archetype grid_1 images for stain '{args.stain}' in {image_folder}"
        )
    excluded = discover_excluded_archetypes(image_folder, args.stain, grid_id=1)
    check_expected_count(archetypes, args.expected_archetypes, f"{args.stain} archetypes", excluded)
    arch_ids = sorted_archetype_ids(archetypes)
    print(f"🧬 {args.stain}: {len(arch_ids)} archetypes -> {', '.join(arch_ids)}")
    if excluded:
        print(f"🚫 excluded by filename prefix: {describe_exclusions(excluded)}")

    llm = make_llm(
        args.model,
        provider=args.provider,
        temp=args.temperature,
        reasoning_effort=args.reasoning_effort,
        verbosity=args.verbosity,
        max_completion_tokens=args.max_completion_tokens,
    )
    if model_supports_temperature(args.model):
        print(f"🧠 Model: {args.model} | temperature={args.temperature}")
    else:
        print(f"🧠 Model: {args.model} | reasoning_effort={args.reasoning_effort} "
              f"| max_completion_tokens={args.max_completion_tokens}")

    written, reused = 0, 0
    for arch_id in arch_ids:
        for grid_id in args.grids:
            required = grid_id in args.required_grids
            image_path = (
                require_grid_image(image_folder, arch_id, args.stain, grid_id)
                if required
                else grid_image_path(image_folder, arch_id, args.stain, grid_id)
            )
            if image_path is None:
                print(f"➖ archetype {arch_id}: no optional grid_{grid_id} image, skipping")
                continue

            out_path = annotation_path(output_folder, arch_id, grid_id)
            if args.resume and out_path.exists() and out_path.stat().st_size > 0:
                print(f"⏩ archetype {arch_id} grid {grid_id}: reusing {out_path.name}")
                reused += 1
                continue

            width, height = verify_image(image_path)
            print(f"🔍 archetype {arch_id} grid {grid_id}: {image_path.name} ({width}x{height})")
            body = analyze_image(
                llm,
                image_path,
                get_annotation_prompt(args.stain),
                context=f"step1 {args.stain} archetype {arch_id} grid {grid_id}",
                require=REQUIRED_SECTIONS,
                detail=args.image_detail,
            )
            write_report(out_path, arch_id, grid_id, image_path, args.model, body)
            print(f"   ✅ {out_path.name} ({len(body)} chars)")
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
    print(f"\n📊 Step 1 complete: {written} written, {reused} reused, {len(arch_ids)} archetypes covered")
    if cost:
        print(f"💸 {cost.report()}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Step 1: per-image archetype annotation")
    parser.add_argument("--stain", required=True)
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--expected_archetypes", type=int, default=0,
                        help="Configured k; 0 disables the count check")
    parser.add_argument("--grids", type=int, nargs="+", default=[1, 2],
                        help="Grid indices to annotate")
    parser.add_argument("--required_grids", type=int, nargs="+", default=[1],
                        help="Grid indices that must exist for every archetype")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider", default=None,
                        help="gateway | azure | auto (default: $LLM_PROVIDER, else auto)")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Ignored by GPT-5 models")
    parser.add_argument("--reasoning_effort", default="low")
    parser.add_argument("--verbosity", default=None)
    parser.add_argument("--max_completion_tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    parser.add_argument("--image_detail", default="high", choices=["low", "high", "auto"])
    parser.add_argument("--resume", action="store_true",
                        help="Skip archetype/grid pairs that already have a non-empty report")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.exit(main(parse_args()))
    except PipelineError as exc:
        print(f"\n❌ Step 1 failed: {exc}", file=sys.stderr)
        sys.exit(1)
