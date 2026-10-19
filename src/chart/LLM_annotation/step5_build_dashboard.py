#!/usr/bin/env python3
"""Step 5 - dashboard.

Renders the step 4 final reports (archetype{i}_final.md) next to their grid
images as one standalone HTML file, with the images base64-embedded so it can
be emailed or opened from anywhere.
"""

import argparse
import base64
import html
import re
import sys
from pathlib import Path

from llm_core import PipelineError
from utils import (
    FINAL_GLOB,
    SECTION_SUMMARY,
    extract_section,
    grid_image_path,
    sorted_archetype_ids,
)

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
.missing { font-style: italic; color: #a00; border: 1px dashed #a00; padding: 0.6em; }
""".strip()


def embed_image(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "jpeg" if suffix in {".jpg", ".jpeg"} else "gif" if suffix == ".gif" else "png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return (f'<img src="data:image/{mime};base64,{encoded}" '
            f'class="full-width" alt="{html.escape(path.name)}"/>')


def build_dashboard(review_dir: Path, image_dir: Path, stain: str, grids, required_grids,
                    require_images: bool):
    reports = sorted(
        review_dir.glob(FINAL_GLOB),
        key=lambda p: int(re.search(r"archetype(\d+)", p.name).group(1)),
    )
    if not reports:
        raise PipelineError(f"No {FINAL_GLOB} reports in {review_dir} - run step 4 first")

    doc = [
        "<html><head><meta charset='UTF-8'>",
        f"<title>Archetype Dashboard - {html.escape(stain)}</title>",
        f"<style>{STYLE}</style></head><body>",
        f"<h1>🧬 Archetype Dashboard - Stain: {html.escape(stain)}</h1><hr>",
        f"<div class='meta'>Stain: {html.escape(stain)} | Reports: {len(reports)}</div>",
    ]

    missing_images, optional_missing = [], []
    rendered = []
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
        for grid_idx in grids:
            img_path = grid_image_path(image_dir, arch_id, stain, grid_idx)
            doc.append("<div class='image-block'>")
            doc.append(f"<div class='meta'>grid_{grid_idx}</div>")
            if img_path is None:
                if grid_idx in required_grids:
                    missing_images.append(f"archetype {arch_id} grid_{grid_idx}")
                else:
                    optional_missing.append(f"archetype {arch_id} grid_{grid_idx}")
                doc.append(f"<p class='missing'>[grid_{grid_idx} image not found]</p>")
            else:
                doc.append(embed_image(img_path))
            doc.append("</div>")
        doc.append("</div>")
        doc.append("<div class='label'>Final Archetype Report:</div>")
        doc.append(f"<div class='report'>{html.escape(report_text)}</div>")
        doc.append("</div>")
        rendered.append(arch_id)

    doc.append("</body></html>")

    if optional_missing:
        print(f"➖ {len(optional_missing)} optional image(s) absent, rendered as placeholders: "
              + "; ".join(optional_missing))
    if missing_images and require_images:
        raise PipelineError(
            "Missing required grid image(s): " + "; ".join(missing_images)
            + "  (pass --allow_missing_images to render anyway)"
        )
    if missing_images:
        print(f"⚠️  {len(missing_images)} required image(s) missing: {'; '.join(missing_images)}")

    return "\n".join(doc), sorted_archetype_ids(rendered)


def main(args) -> int:
    review_dir = Path(args.review_folder)
    image_dir = Path(args.image_folder)
    output_path = Path(args.output)

    if not review_dir.is_dir():
        raise PipelineError(f"Review folder not found: {review_dir}")
    if not image_dir.is_dir():
        raise PipelineError(f"Image folder not found: {image_dir}")

    print(f"📁 Reports: {review_dir.resolve()}")
    print(f"🖼️  Images:  {image_dir.resolve()}")

    content, rendered = build_dashboard(
        review_dir, image_dir, args.stain, args.grids, args.required_grids,
        not args.allow_missing_images,
    )
    if args.expected_archetypes and len(rendered) != args.expected_archetypes:
        raise PipelineError(
            f"Dashboard rendered {len(rendered)} archetypes, expected {args.expected_archetypes}: "
            f"{', '.join(rendered)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    size_mb = output_path.stat().st_size / 1e6
    print(f"\n✅ Dashboard written: {output_path.resolve()} "
          f"({len(rendered)} archetypes, {size_mb:.1f} MB)")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Step 5: build the archetype dashboard")
    parser.add_argument("--stain", required=True)
    parser.add_argument("--review_folder", required=True)
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--grids", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--required_grids", type=int, nargs="+", default=[1],
                        help="Grids whose absence is an error; others render as placeholders")
    parser.add_argument("--expected_archetypes", type=int, default=0)
    parser.add_argument("--allow_missing_images", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.exit(main(parse_args()))
    except PipelineError as exc:
        print(f"\n❌ Step 5 failed: {exc}", file=sys.stderr)
        sys.exit(1)
