#!/usr/bin/env python3
"""Step 4 - contrastive refinement, and the final report.

Archetype summaries from step 3 are embedded and compared; pairs that read too
much alike get a two-image contrastive call in both directions, and each
archetype's summary is then rewritten to fold in what makes it distinct.
Archetypes with no similar partner keep their step 3 summary and say so.

This step owns the final artifact:

    archetype{i}_final.md  =  refined Summary
                            + Proposed Names   (carried over from step 3)
                            + Functional Hypothesis (carried over from step 3)
                            + Refinement Notes (what it was compared against)

The summary is never dropped: if the refinement call cannot produce a usable
one, the step fails rather than write a report without it.
"""

import argparse
import sys
from datetime import datetime
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
from promptlib import build_pairwise_contrast_prompt, build_refinement_prompt
from utils import (
    SECTION_HYPOTHESIS,
    SECTION_NAMES,
    SECTION_REFINEMENT_NOTES,
    SECTION_SUMMARY,
    SYNTHESIS_GLOB,
    check_expected_count,
    compute_pairwise_similarities,
    contrast_filename,
    extract_section,
    final_filename,
    load_contrastive_baselines,
    parse_synthesis_report,
    require_grid_image,
    sorted_archetype_ids,
)

MIN_SUMMARY_CHARS = 150


def run_pairwise_contrasts(llm, pairs, summaries, baselines, image_folder, stain,
                           review_dir, image_detail):
    """Both directions of every flagged pair. Returns arch_id -> [notes]."""
    notes = {}
    total = len(pairs) * 2
    done = 0
    for arch_a, arch_b in pairs:
        img_a = require_grid_image(image_folder, arch_a, stain, 1)
        img_b = require_grid_image(image_folder, arch_b, stain, 1)
        block_a = image_block(img_a, detail=image_detail, max_edge=1024)
        block_b = image_block(img_b, detail=image_detail, max_edge=1024)

        for this_id, other_id, this_block, other_block in (
            (arch_a, arch_b, block_a, block_b),
            (arch_b, arch_a, block_b, block_a),
        ):
            done += 1
            print(f"🔄 [{done}/{total}] contrasting archetype {this_id} against {other_id}...")
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
            print(f"   ✅ {path.name} ({len(text)} chars)")
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


def main(args) -> int:
    review_dir = Path(args.review_folder)
    if not review_dir.is_dir():
        raise PipelineError(f"Review folder not found: {review_dir}")
    image_folder = Path(args.image_folder)
    if not image_folder.is_dir():
        raise PipelineError(f"Image folder not found: {image_folder}")

    print(f"🔌 Provider: {check_llm_credentials(args.provider, args.model)}")

    baselines = load_contrastive_baselines(args.contrastive_json)

    # Sections that survive refinement unchanged.
    sections = {}
    for path in sorted(review_dir.glob(SYNTHESIS_GLOB)):
        arch_id = path.name.split("_")[0].replace("archetype", "")
        sections[arch_id] = parse_synthesis_report(path)
    if not sections:
        raise PipelineError(f"No {SYNTHESIS_GLOB} reports in {review_dir} - run step 3 first")
    check_expected_count(sections, args.expected_archetypes, f"{args.stain} synthesis reports")

    arch_ids = sorted_archetype_ids(sections)
    summaries = {a: sections[a]["summary"] for a in arch_ids}
    print(f"📦 Loaded {len(arch_ids)} synthesis reports: {', '.join(arch_ids)}")

    llm = make_llm(
        args.model,
        provider=args.provider,
        temp=args.temperature,
        reasoning_effort=args.reasoning_effort,
        verbosity=args.verbosity,
        max_completion_tokens=args.max_completion_tokens,
    )

    compared_with = {a: set() for a in arch_ids}
    refined_flag = {a: False for a in arch_ids}

    for iteration in range(1, args.iterations + 1):
        print(f"\n🌀 Iteration {iteration}/{args.iterations}")
        pairs = compute_pairwise_similarities(
            summaries,
            threshold=args.similarity_threshold,
            max_pairs_per_archetype=args.max_pairs_per_archetype,
        )
        if not pairs:
            print("✅ No archetype pairs below the similarity threshold this round.")
            break

        print(f"⚠️  {len(pairs)} similar pair(s) -> {2 * len(pairs)} contrastive calls")
        notes = run_pairwise_contrasts(
            llm, pairs, summaries, baselines, image_folder, args.stain,
            review_dir, args.image_detail,
        )

        for arch_a, arch_b in pairs:
            compared_with[arch_a].add(arch_b)
            compared_with[arch_b].add(arch_a)

        for arch_id in sorted_archetype_ids(notes):
            print(f"🧪 Refining summary for archetype {arch_id}...")
            summaries[arch_id] = refine_summary(
                llm, arch_id, summaries[arch_id], notes[arch_id], args.stain
            )
            refined_flag[arch_id] = True
            print(f"   ✅ {len(summaries[arch_id])} chars")

    combined = []
    for arch_id in arch_ids:
        out_path = review_dir / final_filename(arch_id)
        write_final_report(
            out_path,
            arch_id,
            args.stain,
            args.model,
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
        print(f"📄 {out_path.name} ({'refined' if refined_flag[arch_id] else 'carried over'})")

    header = (
        f"# Final Archetype Reports - {args.stain.upper()}\n"
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with {args.model}_\n\n"
    )
    (review_dir / "combined_archetype_final.md").write_text(
        header + "\n".join(combined), encoding="utf-8"
    )

    n_refined = sum(refined_flag.values())
    print(f"\n✅ Step 4 complete: {len(arch_ids)} final reports "
          f"({n_refined} refined, {len(arch_ids) - n_refined} carried over)")
    cost = cost_callback_of(llm)
    if cost:
        print(f"💸 {cost.report()}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 4: contrastive refinement producing the final archetype reports")
    parser.add_argument("--stain", required=True)
    parser.add_argument("--review_folder", required=True, help="Step 3 output folder")
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--contrastive_json", required=True)
    parser.add_argument("--expected_archetypes", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider", default=None,
                        help="gateway | azure | auto (default: $LLM_PROVIDER, else auto)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--reasoning_effort", default="high")
    parser.add_argument("--verbosity", default=None)
    parser.add_argument("--max_completion_tokens", type=int, default=16000,
                        help="Refinement runs at high reasoning effort and needs a large budget")
    parser.add_argument("--similarity_threshold", type=float, default=0.1,
                        help="Cosine distance below which two summaries are 'too similar'")
    parser.add_argument("--max_pairs_per_archetype", type=int, default=0,
                        help="Keep only the N closest partners per archetype (0 = no cap)")
    parser.add_argument("--image_detail", default="low", choices=["low", "high", "auto"])
    parser.add_argument("--iterations", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.exit(main(parse_args()))
    except PipelineError as exc:
        print(f"\n❌ Step 4 failed: {exc}", file=sys.stderr)
        sys.exit(1)
