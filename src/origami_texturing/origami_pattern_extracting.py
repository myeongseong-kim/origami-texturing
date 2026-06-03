"""OpenAI image helpers for extracting origami pattern."""

import base64
from pathlib import Path

from origami_texturing.config import get_openai_client

ORIGAMI_PATTERN_EXTRACT_PROMPT = """
Transfer only the pattern of the origami in the second image into the white silhouette area of the first image.
Preserve the visual details of the origami's pattern, while removing only crease and shadow. 
Output a PNG-style image with a black background.
""".strip()


def extract_origami_pattern(
    mask_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path = "generated.png",
    model: str = "gpt-image-2",
) -> Path:
    """Extract the origami pattern from the reference image and apply it to the mask image's white area."""

    mask_path = Path(mask_path)
    reference_path = Path(reference_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = get_openai_client()

    with mask_path.open("rb") as mask_file, reference_path.open("rb") as reference_file:
        result = client.images.edit(
            model=model,
            image=[mask_file, reference_file],
            prompt=ORIGAMI_PATTERN_EXTRACT_PROMPT,
            size="1536x1024",
            quality="low",
            output_format="png",
        )

    b64_json = result.data[0].b64_json
    if b64_json is None:
        raise RuntimeError("OpenAI image edit response did not include b64_json image data.")

    image_bytes = base64.b64decode(b64_json)
    output_path.write_bytes(image_bytes)

    return output_path
