#!/usr/bin/env python3
"""Step 3 - synthesis, names and hypothesis.

Text-only. Per archetype: the grid_1 annotation plus the baseline note become a
Summary, then a second call proposes names and one functional hypothesis. This
is the first and only place the pipeline is allowed to interpret biology.

Output: <output_folder>/archetype{i}_synthesis.md
        (step 4 refines these into archetype{i}_final.md)
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from llm_core import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_MODEL,
    PipelineError,
    check_llm_credentials,
    cost_callback_of,
    invoke_text,
    make_llm,
)
from promptlib import build_names_prompt, build_summary_prompt
from utils import (
    SECTION_HYPOTHESIS,
    SECTION_NAMES,
    SECTION_SUMMARY,
    check_expected_count,
    load_contrastive_baselines,
    parse_synthesis_report,
    sorted_archetype_ids,
    synthesis_filename,
)


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


def main(args) -> int:
    report_folder = Path(args.report_folder)
    if not report_folder.is_dir():
        raise PipelineError(f"Step 1 report folder not found: {report_folder}")

    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔌 Provider: {check_llm_credentials(args.provider, args.model)}")

    annotations = load_annotations(report_folder)
    baselines = load_contrastive_baselines(args.contrastive_json)
    check_expected_count(annotations, args.expected_archetypes, f"{args.stain} annotations")

    missing_notes = [a for a in annotations if a not in baselines]
    if missing_notes:
        raise PipelineError(
            "No baseline note for archetype(s): " + ", ".join(sorted_archetype_ids(missing_notes))
        )

    llm_summary = make_llm(
        args.model,
        provider=args.provider,
        temp=args.summary_temp,
        reasoning_effort=args.summary_reasoning_effort,
        verbosity=args.verbosity,
        max_completion_tokens=args.max_completion_tokens,
    )
    llm_names = make_llm(
        args.model,
        provider=args.provider,
        temp=args.names_temp,
        reasoning_effort=args.names_reasoning_effort,
        verbosity=args.verbosity,
        max_completion_tokens=args.max_completion_tokens,
    )

    arch_ids = sorted_archetype_ids(annotations)
    print(f"🧬 Synthesizing {len(arch_ids)} archetypes: {', '.join(arch_ids)}")

    combined = []
    for arch_id in arch_ids:
        annotation = annotations[arch_id]
        baseline_note = baselines.get(arch_id, "")

        print(f"🧠 Archetype {arch_id}: summary...")
        summary_block = invoke_text(
            llm_summary,
            [{"role": "user", "content": build_summary_prompt(arch_id, annotation, baseline_note)}],
            context=f"step3 {args.stain} archetype {arch_id} summary",
            require=(SECTION_SUMMARY,),
        )

        print(f"🧠 Archetype {arch_id}: names and hypothesis...")
        names_block = invoke_text(
            llm_names,
            [{"role": "user", "content": build_names_prompt(arch_id, annotation, args.stain)}],
            context=f"step3 {args.stain} archetype {arch_id} names",
            require=(SECTION_NAMES, SECTION_HYPOTHESIS),
        )

        body = f"{summary_block.strip()}\n\n{names_block.strip()}\n"
        out_path = output_dir / synthesis_filename(arch_id)
        out_path.write_text(body, encoding="utf-8")

        # Fail here rather than let step 4 inherit an unparseable report.
        parse_synthesis_report(out_path)
        print(f"   ✅ {out_path.name}")
        combined.append(f"## Archetype {arch_id}\n\n{body}\n---\n")

    header = (
        f"# Archetype Synthesis - {args.stain.upper()}\n"
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with {args.model}_\n\n"
    )
    (output_dir / "combined_archetype_synthesis.md").write_text(
        header + "\n".join(combined), encoding="utf-8"
    )

    print(f"\n✅ Step 3 complete: {len(arch_ids)} synthesis reports in {output_dir}")
    for llm, label in ((llm_summary, "summary"), (llm_names, "names")):
        cost = cost_callback_of(llm)
        if cost:
            print(f"💸 {label}: {cost.report()}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Step 3: synthesize summaries, names, hypotheses")
    parser.add_argument("--stain", required=True)
    parser.add_argument("--report_folder", required=True, help="Step 1 output folder")
    parser.add_argument("--contrastive_json", required=True, help="Step 2 notes JSON")
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--expected_archetypes", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider", default=None,
                        help="gateway | azure | auto (default: $LLM_PROVIDER, else auto)")
    parser.add_argument("--summary_temp", type=float, default=0.7)
    parser.add_argument("--names_temp", type=float, default=0.2)
    parser.add_argument("--summary_reasoning_effort", default="medium")
    parser.add_argument("--names_reasoning_effort", default="high")
    parser.add_argument("--verbosity", default=None)
    parser.add_argument("--max_completion_tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.exit(main(parse_args()))
    except PipelineError as exc:
        print(f"\n❌ Step 3 failed: {exc}", file=sys.stderr)
        sys.exit(1)
