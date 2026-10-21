# image_utils.py
"""Image loading, encoding and the step 1 vision call.

Missing or unreadable images raise instead of returning a placeholder string:
a grid that cannot be read is a run-stopping problem, not an annotation.
"""

import base64
import io
import logging
import mimetypes
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from llm_core import PipelineError, invoke_text

MIN_RESOLUTION = (512, 512)

ANNOTATION_SYSTEM_PROMPT = (
    "You are a careful microscopy image analyst. When given an image, describe "
    "only what is visible without guessing or bringing in prior biological "
    "knowledge. Use precise and observational language."
)


class ImageError(PipelineError):
    """An input image is missing, unreadable, or unusable."""


def verify_image(image_path) -> Tuple[int, int]:
    """Return (width, height); raise if the file is missing or unreadable."""
    path = Path(image_path)
    if not path.exists():
        raise ImageError(f"Image not found: {path}")
    try:
        with Image.open(path) as img:
            width, height = img.size
    except Exception as exc:
        raise ImageError(f"Could not read image {path}: {exc}") from exc

    if width < MIN_RESOLUTION[0] or height < MIN_RESOLUTION[1]:
        logging.warning(
            "Image %s is below the %sx%s resolution guide (%sx%s)",
            path.name, MIN_RESOLUTION[0], MIN_RESOLUTION[1], width, height,
        )
    return width, height


def encode_image(image_path) -> Tuple[str, str]:
    """Base64-encode an image at full resolution, with its MIME type."""
    path = Path(image_path)
    if not path.exists():
        raise ImageError(f"Image not found: {path}")
    data = path.read_bytes()
    mime_type, _ = mimetypes.guess_type(str(path))
    return base64.b64encode(data).decode(), mime_type or "image/png"


def image_block(image_path, detail: str = "high", max_edge: Optional[int] = None) -> dict:
    """An OpenAI-style image_url content block.

    max_edge downsamples first, which is what the multi-image comparison steps
    use to keep a handful of grids inside one request.
    """
    path = Path(image_path)
    if not path.exists():
        raise ImageError(f"Image not found: {path}")

    if max_edge:
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((max_edge, max_edge))
                buffer = io.BytesIO()
                img.save(buffer, format="PNG", optimize=True)
                encoded = base64.b64encode(buffer.getvalue()).decode()
        except Exception as exc:
            raise ImageError(f"Could not downsample image {path}: {exc}") from exc
        mime_type = "image/png"
    else:
        encoded, mime_type = encode_image(path)

    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}", "detail": detail},
    }


def analyze_image(
    llm,
    image_path,
    prompt: str,
    *,
    context: str,
    require=(),
    detail: str = "high",
) -> str:
    """Send one image plus a prompt and return validated text."""
    messages = [
        {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt.strip()},
                image_block(image_path, detail=detail),
            ],
        },
    ]
    return invoke_text(llm, messages, context=context, require=require)
