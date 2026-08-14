"""Integrated end-to-end origami texturing pipeline.

The tested processing from notebooks 01-04 is implemented as normal Python
stages. Images are saved to disk and reported through logs; no image windows
are opened.
"""

from __future__ import annotations

import base64
import copy
import gc
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageOps
from sklearn.cluster import KMeans
from tqdm import tqdm

from origami_texturing import get_openai_client
from origami_texturing.paths import (
    EXTERNAL_DIR,
    INPUTS_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    ensure_project_dirs,
)

LOGGER = logging.getLogger("origami_texturing_pipeline")
POSE_NAMES = ("wingup", "wingdown")

# Stage 01
REFERENCE_IMAGE_SIZE = (1536, 2048)
IMAGE_MODEL = "gpt-image-2"
IMAGE_OUTPUT_SIZE = "1536x1024"
IMAGE_QUALITY = "low"
ORIGAMI_PATTERN_EXTRACT_PROMPT = """
Transfer only the pattern from the origami in the pair image into the white silhouette area of the mask-pair image.
Preserve colors, textures, and visual details of the origami's pattern, while removing only shadow and shade.
Return a PNG-style image with a black background.
""".strip()

# Stage 02
PATCH_SIZE = 16
DINO_IMAGE_SIZE = 1024
MASK_FG_THRESHOLD = 0.5
VERTEX_MASK_FG_THRESHOLD = 0.05
SIMILARITY_TEMPERATURE = 0.1
BIAS_ALPHA = 1.0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINO_MODEL_NAME = "dinov3_vith16plus"
DINO_MODEL_LAYERS = 32
DINO_WEIGHTS_PATH = MODELS_DIR / "dinov3" / "dinov3_vith16plus.pth"
CONTOUR_ALPHA_THRESHOLD = 0.5
CONTOUR_APPROXIMATION_EPSILON_RATIO = 0.012
CONTOUR_MIN_AREA = 100.0

# Stage 04
REGION_FACES = {
    "head": ["head"],  # The beak is intentionally excluded.
    "body": ["belly", "chest"],
    "wing": ["wing", "lower-wing", "upper-wing"],
    "tail": ["tail"],
}
N_CANDIDATES_PER_REGION = 3
ALPHA_THRESHOLD = 128
MAX_SAMPLES = 100_000
RANDOM_STATE = 42


def configure_logging() -> None:
    """Configure concise console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def reset_runtime_memory(phase: str) -> None:
    """Collect Python objects and release unused CUDA cache."""
    collected_objects = gc.collect()
    cuda_status = "unavailable"
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            cuda_status = "cleared"
    except Exception as error:  # Cleanup must not hide the pipeline result.
        cuda_status = f"failed ({error})"
    LOGGER.info(
        "Memory reset (%s): collected %d Python objects; CUDA cache=%s",
        phase,
        collected_objects,
        cuda_status,
    )


def require_file(path: Path, label: str) -> None:
    """Raise an actionable error for a missing pipeline input."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_json(path: Path) -> dict:
    """Load a UTF-8 JSON object."""
    require_file(path, "JSON")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


# -----------------------------------------------------------------------------
# Stage 01: prepare references and generate pattern images
# -----------------------------------------------------------------------------


def load_reference_image(path: Path) -> Image.Image:
    """Load, fill-resize, and center-crop a reference image."""
    require_file(path, "Reference image")
    with Image.open(path) as image:
        return ImageOps.fit(
            image.convert("RGBA"),
            REFERENCE_IMAGE_SIZE,
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )


def generate_origami_pattern(
    mask_path: Path,
    reference_path: Path,
    output_path: Path,
) -> Path:
    """Transfer the reference pattern with the OpenAI Images API."""
    require_file(mask_path, "Pair mask")
    require_file(reference_path, "Resized reference pair")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = get_openai_client()

    with mask_path.open("rb") as mask_file, reference_path.open("rb") as reference_file:
        result = client.images.edit(
            model=IMAGE_MODEL,
            image=[mask_file, reference_file],
            prompt=ORIGAMI_PATTERN_EXTRACT_PROMPT,
            size=IMAGE_OUTPUT_SIZE,
            quality=IMAGE_QUALITY,
            output_format="png",
        )

    encoded_image = result.data[0].b64_json
    if encoded_image is None:
        raise RuntimeError("OpenAI image edit response did not include image data.")
    output_path.write_bytes(base64.b64decode(encoded_image))
    return output_path


def stage_01_generate_images() -> None:
    """Resize references, call the image model, split output, and remove backgrounds."""
    from rembg import new_session, remove

    LOGGER.info("Stage 01/04: preparing references and generating pattern images")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    input_paths = {
        "wingup": INPUTS_DIR / "wingup.png",
        "wingdown": INPUTS_DIR / "wingdown.png",
    }
    resized_paths = {
        pose_name: PROCESSED_DATA_DIR / f"resized-{pose_name}.png"
        for pose_name in POSE_NAMES
    }
    resized_images = {
        pose_name: load_reference_image(input_paths[pose_name])
        for pose_name in POSE_NAMES
    }

    for pose_name in POSE_NAMES:
        resized_images[pose_name].save(resized_paths[pose_name])
        LOGGER.info("Saved resized %s image to: %s", pose_name, resized_paths[pose_name])

    pair = Image.new(
        "RGBA",
        (REFERENCE_IMAGE_SIZE[0] * 2, REFERENCE_IMAGE_SIZE[1]),
        (255, 255, 255, 0),
    )
    pair.paste(resized_images["wingup"], (0, 0), resized_images["wingup"])
    pair.paste(
        resized_images["wingdown"],
        (REFERENCE_IMAGE_SIZE[0], 0),
        resized_images["wingdown"],
    )
    resized_pair_path = PROCESSED_DATA_DIR / "resized-pair.png"
    pair.save(resized_pair_path)
    LOGGER.info("Saved resized pair image to: %s", resized_pair_path)

    generated_pair_path = PROCESSED_DATA_DIR / "generated-pair.png"
    generate_origami_pattern(
        mask_path=RAW_DATA_DIR / "dove-pair-mask.png",
        reference_path=resized_pair_path,
        output_path=generated_pair_path,
    )
    LOGGER.info("Saved generated pair image to: %s", generated_pair_path)

    with Image.open(generated_pair_path) as source:
        generated_pair = source.convert("RGBA")
        midpoint = generated_pair.width // 2
        raw_images = {
            "wingup": generated_pair.crop((0, 0, midpoint, generated_pair.height)),
            "wingdown": generated_pair.crop(
                (midpoint, 0, generated_pair.width, generated_pair.height)
            ),
        }

    # Preserve the tested notebook filename for compatibility.
    generated_paths = {
        "wingup": PROCESSED_DATA_DIR / "generated-wingup.png",
        "wingdown": PROCESSED_DATA_DIR / "generated-wingdwon.png",
    }
    rembg_session = new_session()
    try:
        for pose_name in POSE_NAMES:
            generated = remove(raw_images[pose_name], session=rembg_session).convert("RGBA")
            generated.save(generated_paths[pose_name])
            LOGGER.info(
                "Saved background-removed %s image to: %s",
                pose_name,
                generated_paths[pose_name],
            )
    finally:
        del rembg_session


# -----------------------------------------------------------------------------
# Stage 02: match DINOv3 features, snap vertices, and warp patterns
# -----------------------------------------------------------------------------


def load_dino_model() -> torch.nn.Module:
    """Load the configured DINOv3 model and local weights on CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 02 requires a CUDA-enabled PyTorch installation and GPU.")
    require_file(DINO_WEIGHTS_PATH, "DINOv3 weights")

    local_repository = EXTERNAL_DIR / "dinov3"
    configured_location = os.getenv("DINOV3_LOCATION")
    if configured_location:
        repository = configured_location
        source = "local" if Path(configured_location).exists() else "github"
    elif local_repository.exists():
        repository = str(local_repository)
        source = "local"
    else:
        repository = "facebookresearch/dinov3"
        source = "github"

    LOGGER.info("Loading DINOv3 from: %s", repository)
    model = torch.hub.load(
        repo_or_dir=repository,
        model=DINO_MODEL_NAME,
        source=source,
        weights=str(DINO_WEIGHTS_PATH),
    )
    return model.cuda().eval()


def load_image_item(path: Path, alpha_threshold: int = 0) -> dict:
    """Load RGB content, alpha values, and a foreground mask."""
    require_file(path, "Image")
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"))
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return {
        "image": Image.alpha_composite(background, rgba).convert("RGB"),
        "alpha": alpha,
        "mask": alpha > alpha_threshold,
    }


def resize_for_dino(
    image: Image.Image,
    image_size: int = DINO_IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    """Resize to a patch-aligned height while preserving aspect ratio."""
    width, height = image.size
    height_patches = image_size // patch_size
    width_patches = int((width * image_size) / (height * patch_size))
    size = (height_patches * patch_size, width_patches * patch_size)
    return TF.to_tensor(TF.resize(image, size))


def compute_patch_matches(pair: dict, model: torch.nn.Module, pose_name: str) -> dict:
    """Compute dense DINOv3 target-to-pattern patch correspondences."""
    quantizer = torch.nn.Conv2d(1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False)
    quantizer.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))
    quantizer.requires_grad_(False)
    patch_masks = []
    patch_features = []

    with torch.inference_mode():
        for image_kind in tqdm(("target", "pattern"), desc=f"DINOv3 {pose_name}"):
            item = pair[image_kind]
            mask_image = Image.fromarray(item["mask"].astype(np.uint8) * 255, mode="L")
            patch_masks.append(quantizer(resize_for_dino(mask_image)).squeeze().cpu())

            image_tensor = resize_for_dino(item["image"].convert("RGB"))
            image_tensor = TF.normalize(
                image_tensor,
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ).unsqueeze(0).cuda()
            intermediate = model.get_intermediate_layers(
                image_tensor,
                n=range(DINO_MODEL_LAYERS),
                reshape=True,
                norm=True,
            )
            patch_features.append(intermediate[-1].squeeze().cpu())

    target_features = F.normalize(patch_features[0], p=2, dim=0)
    pattern_features = F.normalize(patch_features[1], p=2, dim=0)
    feature_dimension = target_features.shape[0]
    heatmaps = torch.einsum(
        "k f, f h w -> k h w",
        target_features.view(feature_dimension, -1).permute(1, 0),
        pattern_features,
    )
    pattern_indices = torch.flatten(heatmaps, start_dim=-2).argmax(dim=-1)
    match_scores = torch.flatten(heatmaps, start_dim=-2).max(dim=-1).values
    target_foreground = patch_masks[0].view(-1) > MASK_FG_THRESHOLD
    pattern_foreground = patch_masks[1].view(-1)[pattern_indices] > MASK_FG_THRESHOLD
    foreground_matches = target_foreground & pattern_foreground
    LOGGER.info(
        "%s: matched %d foreground patches",
        pose_name,
        int(foreground_matches.sum().item()),
    )
    return {
        "patch_features": [target_features, pattern_features],
        "patch_masks": patch_masks,
        "pattern_indices": pattern_indices,
        "match_scores": match_scores,
    }


def patch_center_to_original_xy(row: int, column: int, scale: float) -> list[float]:
    """Convert a DINOv3 patch center to original-image coordinates."""
    return [
        float((column + 0.5) * PATCH_SIZE * scale),
        float((row + 0.5) * PATCH_SIZE * scale),
    ]


def get_neighboring_patch_indices(
    vertex_xy: list[float],
    scale: float,
    height_patches: int,
    width_patches: int,
) -> list[dict]:
    """Return valid patch cells immediately surrounding a topology vertex."""
    resized_x = vertex_xy[0] / scale
    resized_y = vertex_xy[1] / scale
    left_column = int(np.floor(resized_x / PATCH_SIZE - 0.5))
    top_row = int(np.floor(resized_y / PATCH_SIZE - 0.5))
    neighbors = (
        (top_row, left_column),
        (top_row, left_column + 1),
        (top_row + 1, left_column),
        (top_row + 1, left_column + 1),
    )
    return [
        {"row": row, "column": column, "index": row * width_patches + column}
        for row, column in neighbors
        if 0 <= row < height_patches and 0 <= column < width_patches
    ]


def weighted_pattern_vertex(
    vertex_xy: list[float],
    matches: dict,
    target_scale: float,
    pattern_scale: float,
) -> list[float] | None:
    """Estimate and bias-correct a matching pattern vertex."""
    target_features, pattern_features = matches["patch_features"]
    target_height, target_width = target_features.shape[1:]
    pattern_width = pattern_features.shape[2]
    candidates = []

    for patch in get_neighboring_patch_indices(
        vertex_xy,
        target_scale,
        target_height,
        target_width,
    ):
        target_index = patch["index"]
        pattern_index = int(matches["pattern_indices"][target_index].item())
        pattern_row = pattern_index // pattern_width
        pattern_column = pattern_index % pattern_width
        target_mask = float(matches["patch_masks"][0].view(-1)[target_index].item())
        pattern_mask = float(matches["patch_masks"][1].view(-1)[pattern_index].item())
        if target_mask <= VERTEX_MASK_FG_THRESHOLD or pattern_mask <= VERTEX_MASK_FG_THRESHOLD:
            continue
        candidates.append(
            {
                "target_xy": patch_center_to_original_xy(
                    patch["row"], patch["column"], target_scale
                ),
                "pattern_xy": patch_center_to_original_xy(
                    pattern_row, pattern_column, pattern_scale
                ),
                "similarity": float(matches["match_scores"][target_index].item()),
            }
        )

    if not candidates:
        return None
    scores = torch.tensor([item["similarity"] for item in candidates], dtype=torch.float32)
    weights = torch.softmax(scores / SIMILARITY_TEMPERATURE, dim=0).numpy()
    target_points = np.asarray([item["target_xy"] for item in candidates], dtype=np.float64)
    pattern_points = np.asarray(
        [item["pattern_xy"] for item in candidates], dtype=np.float64
    )
    weighted_target = (target_points * weights[:, None]).sum(axis=0)
    weighted_pattern = (pattern_points * weights[:, None]).sum(axis=0)
    target_bias = np.asarray(vertex_xy, dtype=np.float64) - weighted_target
    pattern_bias = target_bias * (pattern_scale / target_scale)
    return (weighted_pattern + BIAS_ALPHA * pattern_bias).round(3).tolist()


def match_topology_vertices(pair: dict, topology: dict, matches: dict) -> dict:
    """Match all named topology vertices into pattern-image coordinates."""
    target_scale = pair["target"]["image"].height / DINO_IMAGE_SIZE
    pattern_scale = pair["pattern"]["image"].height / DINO_IMAGE_SIZE
    return {
        vertex_name: {
            "target_xy": [float(vertex_xy[0]), float(vertex_xy[1])],
            "pattern_xy": weighted_pattern_vertex(
                vertex_xy,
                matches,
                target_scale,
                pattern_scale,
            ),
        }
        for vertex_name, vertex_xy in topology["vertices"].items()
    }


def contour_mask_from_alpha(alpha: np.ndarray) -> np.ndarray:
    """Convert an alpha channel into a binary OpenCV mask."""
    alpha = np.asarray(alpha)
    if alpha.dtype == np.bool_:
        mask = alpha
    elif np.issubdtype(alpha.dtype, np.integer):
        mask = alpha > int(round(CONTOUR_ALPHA_THRESHOLD * 255))
    else:
        mask = alpha > CONTOUR_ALPHA_THRESHOLD
    return mask.astype(np.uint8) * 255


def extract_contours_xy(alpha: np.ndarray, chain_mode: int) -> list[np.ndarray]:
    """Extract valid external contours with the requested chain mode."""
    contours, _ = cv2.findContours(
        contour_mask_from_alpha(alpha),
        cv2.RETR_EXTERNAL,
        chain_mode,
    )
    return [
        contour.reshape(-1, 2).astype(float)
        for contour in contours
        if cv2.contourArea(contour) >= CONTOUR_MIN_AREA
    ]


def extract_contour_points_xy(alpha: np.ndarray) -> np.ndarray:
    """Extract all points from valid external contours."""
    points = extract_contours_xy(alpha, cv2.CHAIN_APPROX_NONE)
    return np.vstack(points) if points else np.empty((0, 2), dtype=float)


def extract_contour_vertices_xy(alpha: np.ndarray) -> np.ndarray:
    """Approximate valid contours as unique polygon vertices."""
    vertices = []
    for contour_xy in extract_contours_xy(alpha, cv2.CHAIN_APPROX_SIMPLE):
        contour = contour_xy.reshape(-1, 1, 2).astype(np.float32)
        epsilon = CONTOUR_APPROXIMATION_EPSILON_RATIO * cv2.arcLength(contour, True)
        approximated = cv2.approxPolyDP(contour, epsilon, True)
        vertices.extend(approximated.reshape(-1, 2).astype(float))
    if not vertices:
        return np.empty((0, 2), dtype=float)
    return np.unique(np.round(np.asarray(vertices), 2), axis=0)


def snap_points_to_nearest_vertices(
    points_xy: np.ndarray,
    vertices_xy: np.ndarray,
    fallback_contour_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Snap points uniquely to vertices and otherwise to nearest contour points."""
    if len(points_xy) == 0:
        empty = np.zeros(0, dtype=bool)
        return points_xy.copy(), empty, empty

    snapped_points = points_xy.copy()
    snapped_mask = np.zeros(len(points_xy), dtype=bool)
    vertex_snapped_mask = np.zeros(len(points_xy), dtype=bool)
    if len(vertices_xy) > 0:
        distances = np.linalg.norm(points_xy[:, None, :] - vertices_xy[None, :, :], axis=2)
        preferences = np.argsort(distances, axis=1)
        next_preferences = np.zeros(len(points_xy), dtype=int)
        point_by_vertex: dict[int, int] = {}
        vertex_by_point = np.full(len(points_xy), -1, dtype=int)
        pending = list(range(len(points_xy)))

        while pending:
            point_index = pending.pop(0)
            if next_preferences[point_index] >= len(vertices_xy):
                continue
            vertex_index = int(preferences[point_index, next_preferences[point_index]])
            next_preferences[point_index] += 1
            current_point = point_by_vertex.get(vertex_index)
            if current_point is None:
                point_by_vertex[vertex_index] = point_index
                vertex_by_point[point_index] = vertex_index
                continue
            new_distance = distances[point_index, vertex_index]
            current_distance = distances[current_point, vertex_index]
            if new_distance < current_distance or (
                new_distance == current_distance and point_index < current_point
            ):
                point_by_vertex[vertex_index] = point_index
                vertex_by_point[point_index] = vertex_index
                vertex_by_point[current_point] = -1
                pending.append(current_point)
            else:
                pending.append(point_index)

        vertex_snapped_mask = vertex_by_point >= 0
        point_indices = np.where(vertex_snapped_mask)[0]
        if len(point_indices) > 0:
            snapped_points[point_indices] = vertices_xy[vertex_by_point[point_indices]]
            snapped_mask[point_indices] = True

    fallback_mask = ~snapped_mask
    if len(fallback_contour_xy) > 0 and fallback_mask.any():
        point_indices = np.where(fallback_mask)[0]
        distances = np.linalg.norm(
            points_xy[point_indices, None, :] - fallback_contour_xy[None, :, :],
            axis=2,
        )
        nearest_indices = np.argmin(distances, axis=1)
        snapped_points[point_indices] = fallback_contour_xy[nearest_indices]
        snapped_mask[point_indices] = True
    return snapped_points, snapped_mask, vertex_snapped_mask


def snap_matched_vertices(matched_vertices: dict, pattern_alpha: np.ndarray) -> dict:
    """Snap matched pattern vertices onto the extracted pattern contour."""
    snapped_vertices = copy.deepcopy(matched_vertices)
    valid_names = [
        name for name, vertex in matched_vertices.items() if vertex["pattern_xy"] is not None
    ]
    raw_points = np.asarray(
        [matched_vertices[name]["pattern_xy"] for name in valid_names],
        dtype=float,
    )
    contour_vertices = extract_contour_vertices_xy(pattern_alpha)
    contour_points = extract_contour_points_xy(pattern_alpha)
    snapped_points, snapped_mask, vertex_mask = snap_points_to_nearest_vertices(
        raw_points,
        contour_vertices,
        contour_points,
    )
    for vertex_name, snapped_xy in zip(valid_names, snapped_points):
        snapped_vertices[vertex_name]["pattern_xy"] = snapped_xy.round(3).tolist()
    LOGGER.info(
        "Snapped %d/%d vertices (%d polygon vertices, %d contour fallbacks)",
        int(snapped_mask.sum()),
        len(snapped_mask),
        int(vertex_mask.sum()),
        int((snapped_mask & ~vertex_mask).sum()),
    )
    return snapped_vertices


def affine_warp_triangle_rgba(
    source_rgba: np.ndarray,
    output_size: tuple[int, int],
    source_triangle: np.ndarray,
    target_triangle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp one RGBA triangle and return it with its destination mask."""
    output_width, output_height = output_size
    transform = cv2.getAffineTransform(
        source_triangle.astype(np.float32),
        target_triangle.astype(np.float32),
    )
    warped = cv2.warpAffine(
        source_rgba,
        transform,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    triangle_mask = np.zeros((output_height, output_width), dtype=np.uint8)
    cv2.fillConvexPoly(triangle_mask, np.round(target_triangle).astype(np.int32), 255)
    return warped, triangle_mask


def alpha_composite_rgba(base_rgba: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    """Alpha-composite two uint8 RGBA arrays."""
    base = base_rgba.astype(np.float32) / 255.0
    overlay = overlay_rgba.astype(np.float32) / 255.0
    overlay_alpha = overlay[..., 3:4]
    base_alpha = base[..., 3:4]
    output_alpha = overlay_alpha + base_alpha * (1.0 - overlay_alpha)
    output_rgb = np.zeros_like(base[..., :3])
    nonzero = output_alpha[..., 0] > 1e-6
    output_rgb[nonzero] = (
        overlay[..., :3][nonzero] * overlay_alpha[..., 0:1][nonzero]
        + base[..., :3][nonzero]
        * base_alpha[..., 0:1][nonzero]
        * (1.0 - overlay_alpha[..., 0:1][nonzero])
    ) / output_alpha[..., 0:1][nonzero]
    return np.clip(
        np.dstack([output_rgb, output_alpha[..., 0]]) * 255.0,
        0,
        255,
    ).astype(np.uint8)


def warp_pattern_to_topology(
    pattern_rgba: np.ndarray,
    target_size: tuple[int, int],
    topology: dict,
    matched_vertices: dict,
) -> tuple[np.ndarray, int]:
    """Warp all valid triangular pattern faces into topology coordinates."""
    output_width, output_height = target_size
    warped_output = np.zeros((output_height, output_width, 4), dtype=np.uint8)
    warped_count = 0
    for face_name, vertex_ids in topology["faces"].items():
        if len(vertex_ids) != 3:
            LOGGER.warning("Skipping %s: expected three vertices", face_name)
            continue
        missing = [
            vertex_id
            for vertex_id in vertex_ids
            if vertex_id not in matched_vertices
            or matched_vertices[vertex_id]["pattern_xy"] is None
        ]
        if missing:
            LOGGER.warning("Skipping %s: missing pattern matches %s", face_name, missing)
            continue
        source_triangle = np.asarray(
            [matched_vertices[vertex_id]["pattern_xy"] for vertex_id in vertex_ids],
            dtype=np.float32,
        )
        target_triangle = np.asarray(
            [topology["vertices"][vertex_id] for vertex_id in vertex_ids],
            dtype=np.float32,
        )
        if abs(cv2.contourArea(source_triangle.reshape(-1, 1, 2))) < 1.0:
            LOGGER.warning("Skipping %s: degenerate source triangle", face_name)
            continue
        if abs(cv2.contourArea(target_triangle.reshape(-1, 1, 2))) < 1.0:
            LOGGER.warning("Skipping %s: degenerate target triangle", face_name)
            continue
        warped_face, triangle_mask = affine_warp_triangle_rgba(
            pattern_rgba,
            target_size,
            source_triangle,
            target_triangle,
        )
        warped_face[..., 3] = np.minimum(warped_face[..., 3], triangle_mask)
        warped_output = alpha_composite_rgba(warped_output, warped_face)
        warped_count += 1
    return warped_output, warped_count


def stage_02_warp_patterns() -> None:
    """Match patterns to topology, snap vertices, and save warped images."""
    LOGGER.info("Stage 02/04: matching DINOv3 features and warping patterns")
    topology_paths = {
        pose_name: RAW_DATA_DIR / f"dove-{pose_name}-topology.json"
        for pose_name in POSE_NAMES
    }
    target_paths = {
        pose_name: RAW_DATA_DIR / f"dove-{pose_name}-target.png"
        for pose_name in POSE_NAMES
    }
    generated_paths = {
        "wingup": PROCESSED_DATA_DIR / "generated-wingup.png",
        "wingdown": PROCESSED_DATA_DIR / "generated-wingdwon.png",
    }
    warped_paths = {
        pose_name: PROCESSED_DATA_DIR / f"warped-{pose_name}.png"
        for pose_name in POSE_NAMES
    }
    pairs = {
        pose_name: {
            "target": load_image_item(target_paths[pose_name]),
            "pattern": load_image_item(generated_paths[pose_name]),
        }
        for pose_name in POSE_NAMES
    }
    topologies = {
        pose_name: load_json(topology_paths[pose_name])
        for pose_name in POSE_NAMES
    }

    model = load_dino_model()
    matched_vertices_by_pose = {}
    try:
        for pose_name in POSE_NAMES:
            matches = compute_patch_matches(pairs[pose_name], model, pose_name)
            matched_vertices = match_topology_vertices(
                pairs[pose_name],
                topologies[pose_name],
                matches,
            )
            matched_vertices_by_pose[pose_name] = matched_vertices
            matched_count = sum(
                vertex["pattern_xy"] is not None for vertex in matched_vertices.values()
            )
            LOGGER.info(
                "%s: matched %d/%d topology vertices",
                pose_name,
                matched_count,
                len(matched_vertices),
            )
            del matches
    finally:
        del model
        reset_runtime_memory("stage 02 DINOv3 inference")

    for pose_name in POSE_NAMES:
        snapped_vertices = snap_matched_vertices(
            matched_vertices_by_pose[pose_name],
            pairs[pose_name]["pattern"]["alpha"],
        )
        with Image.open(generated_paths[pose_name]) as pattern_image:
            pattern_rgba = np.asarray(pattern_image.convert("RGBA"))
        warped_rgba, warped_count = warp_pattern_to_topology(
            pattern_rgba,
            pairs[pose_name]["target"]["image"].size,
            topologies[pose_name],
            snapped_vertices,
        )
        Image.fromarray(warped_rgba, mode="RGBA").save(warped_paths[pose_name])
        LOGGER.info(
            "Saved %s warped image with %d/%d faces to: %s",
            pose_name,
            warped_count,
            len(topologies[pose_name]["faces"]),
            warped_paths[pose_name],
        )


# -----------------------------------------------------------------------------
# Stage 03: map topology images to the flat-net UV texture
# -----------------------------------------------------------------------------


def collect_points(data: dict) -> dict[str, np.ndarray]:
    """Collect every named point category used by faces."""
    points = {}
    for key in ("vertices", "features", "quaters", "quarters"):
        points.update(data.get(key, {}))
    return {name: np.asarray(xy, dtype=np.float32) for name, xy in points.items()}


def resolve_faces(data: dict) -> dict[str, np.ndarray]:
    """Resolve named triangular faces to arrays of XY coordinates."""
    points = collect_points(data)
    faces = {}
    for face_name, point_names in data["faces"].items():
        missing = [point_name for point_name in point_names if point_name not in points]
        if missing:
            raise KeyError(f"{face_name} is missing points: {missing}")
        if len(point_names) != 3:
            raise ValueError(f"{face_name} must have exactly three points")
        faces[face_name] = np.asarray(
            [points[point_name] for point_name in point_names],
            dtype=np.float32,
        )
    return faces


def make_decalcomanie_y_equals_x(rgba: np.ndarray) -> np.ndarray:
    """Merge an RGBA UV texture with its reflection across y=x."""
    if rgba.shape[0] != rgba.shape[1]:
        raise ValueError("y=x decalcomanie requires a square UV image")
    rgba_float = rgba.astype(np.float32)
    mirrored = np.transpose(rgba_float, (1, 0, 2))
    alpha = rgba_float[..., 3:4] / 255.0
    mirrored_alpha = mirrored[..., 3:4] / 255.0
    weight_sum = alpha + mirrored_alpha
    rgb = np.divide(
        rgba_float[..., :3] * alpha + mirrored[..., :3] * mirrored_alpha,
        np.maximum(weight_sum, 1e-6),
    )
    output_alpha = np.maximum(alpha, mirrored_alpha) * 255.0
    return np.clip(
        np.concatenate([rgb, output_alpha], axis=2),
        0,
        255,
    ).astype(np.uint8)


def map_topology_image_to_net_uv(
    topology_image: Image.Image,
    topology: dict,
    net: dict,
    topology_faces: dict[str, np.ndarray],
    net_faces: dict[str, np.ndarray],
) -> tuple[Image.Image, int]:
    """Affine-map topology faces to the flat net and enforce y=x symmetry."""
    source_rgba = np.asarray(topology_image.convert("RGBA"), dtype=np.uint8)
    source_height, source_width = source_rgba.shape[:2]
    topology_width, topology_height = topology["resolution"]
    net_width, net_height = net["resolution"]
    source_scale = np.asarray(
        [source_width / topology_width, source_height / topology_height],
        dtype=np.float32,
    )
    uv_rgba = np.zeros((net_height, net_width, 4), dtype=np.uint8)
    mapped_count = 0
    for face_name, topology_triangle in topology_faces.items():
        source_triangle = topology_triangle * source_scale
        destination_triangle = net_faces[face_name]
        transform = cv2.getAffineTransform(source_triangle, destination_triangle)
        warped_face = cv2.warpAffine(
            source_rgba,
            transform,
            (net_width, net_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        face_mask = np.zeros((net_height, net_width), dtype=np.uint8)
        cv2.fillConvexPoly(
            face_mask,
            np.round(destination_triangle).astype(np.int32),
            255,
        )
        pixels = face_mask > 0
        uv_rgba[pixels] = warped_face[pixels]
        mapped_count += 1
    symmetric = make_decalcomanie_y_equals_x(uv_rgba)
    return Image.fromarray(symmetric, mode="RGBA"), mapped_count


def make_symmetric_face_mask(
    face_names: list[str],
    net_faces: dict[str, np.ndarray],
    size: tuple[int, int],
) -> np.ndarray:
    """Build a y=x-symmetric mask for selected flat-net faces."""
    width, height = size
    if width != height:
        raise ValueError("The y=x symmetric UV texture must be square")
    mask = np.zeros((height, width), dtype=np.uint8)
    for face_name in face_names:
        if face_name not in net_faces:
            raise KeyError(f"Net face not found: {face_name}")
        cv2.fillConvexPoly(
            mask,
            np.round(net_faces[face_name]).astype(np.int32),
            255,
        )
    boolean_mask = mask > 0
    return boolean_mask | boolean_mask.T


def stage_03_map_texture() -> None:
    """Map warped poses to UV space and save the composed texture."""
    LOGGER.info("Stage 03/04: mapping topology images to flat-net UV texture")
    topology_paths = {
        pose_name: RAW_DATA_DIR / f"dove-{pose_name}-topology.json"
        for pose_name in POSE_NAMES
    }
    net_paths = {
        pose_name: RAW_DATA_DIR / f"dove-{pose_name}-net.json"
        for pose_name in POSE_NAMES
    }
    warped_paths = {
        pose_name: PROCESSED_DATA_DIR / f"warped-{pose_name}.png"
        for pose_name in POSE_NAMES
    }
    uv_paths = {
        pose_name: PROCESSED_DATA_DIR / f"uv-{pose_name}.png"
        for pose_name in POSE_NAMES
    }

    face_data = {}
    uv_images = {}
    for pose_name in POSE_NAMES:
        topology = load_json(topology_paths[pose_name])
        net = load_json(net_paths[pose_name])
        topology_faces = resolve_faces(topology)
        net_faces = resolve_faces(net)
        missing_faces = sorted(set(topology_faces) - set(net_faces))
        if missing_faces:
            raise KeyError(f"{pose_name} net is missing topology faces: {missing_faces}")
        require_file(warped_paths[pose_name], "Warped image")
        with Image.open(warped_paths[pose_name]) as warped_image:
            uv_image, mapped_count = map_topology_image_to_net_uv(
                warped_image.convert("RGBA"),
                topology,
                net,
                topology_faces,
                net_faces,
            )
        uv_image.save(uv_paths[pose_name])
        LOGGER.info(
            "Saved %s UV image with %d mapped faces to: %s",
            pose_name,
            mapped_count,
            uv_paths[pose_name],
        )
        face_data[pose_name] = {"net_faces": net_faces}
        uv_images[pose_name] = uv_image

    wingup_uv = uv_images["wingup"].convert("RGBA")
    wingdown_uv = uv_images["wingdown"].convert("RGBA")
    if wingup_uv.size != wingdown_uv.size:
        raise ValueError(f"UV sizes must match: {wingup_uv.size} != {wingdown_uv.size}")
    wingup_rgba = np.asarray(wingup_uv, dtype=np.uint8)
    wingdown_rgba = np.asarray(wingdown_uv, dtype=np.uint8)
    overlap_mask = (wingup_rgba[..., 3] > 0) & (wingdown_rgba[..., 3] > 0)
    head_beak_mask = make_symmetric_face_mask(
        ["head", "beak"],
        face_data["wingdown"]["net_faces"],
        wingdown_uv.size,
    )
    tail_mask = make_symmetric_face_mask(
        ["tail"],
        face_data["wingup"]["net_faces"],
        wingup_uv.size,
    )
    wingdown_on_top = np.asarray(
        Image.alpha_composite(wingup_uv, wingdown_uv),
        dtype=np.uint8,
    ).copy()
    wingup_on_top = np.asarray(
        Image.alpha_composite(wingdown_uv, wingup_uv),
        dtype=np.uint8,
    )
    texture_rgba = wingdown_on_top
    head_beak_overlap = overlap_mask & head_beak_mask
    tail_overlap = overlap_mask & tail_mask
    texture_rgba[head_beak_overlap] = wingdown_on_top[head_beak_overlap]
    texture_rgba[tail_overlap] = wingup_on_top[tail_overlap]
    texture_path = OUTPUT_DIR / "texture.png"
    Image.fromarray(texture_rgba, mode="RGBA").save(texture_path)
    LOGGER.info(
        "Saved final texture to: %s (head/beak overlap=%d, tail overlap=%d)",
        texture_path,
        int(head_beak_overlap.sum()),
        int(tail_overlap.sum()),
    )


# -----------------------------------------------------------------------------
# Stage 04: extract dominant colors by anatomical region
# -----------------------------------------------------------------------------


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 or float sRGB values in [0, 255] to CIELAB (D65)."""
    srgb = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    rgb_to_xyz = np.asarray(
        [
            [0.4124564, 0.2126729, 0.0193339],
            [0.3575761, 0.7151522, 0.1191920],
            [0.1804375, 0.0721750, 0.9503041],
        ]
    )
    xyz = linear @ rgb_to_xyz
    xyz /= np.asarray([0.95047, 1.0, 1.08883])
    epsilon = 216 / 24389
    kappa = 24389 / 27
    transformed = np.where(xyz > epsilon, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    return np.column_stack(
        (
            116 * transformed[:, 1] - 16,
            500 * (transformed[:, 0] - transformed[:, 1]),
            200 * (transformed[:, 1] - transformed[:, 2]),
        )
    )


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIELAB (D65) values to clipped uint8 sRGB values."""
    lab = np.atleast_2d(np.asarray(lab, dtype=np.float64))
    fy = (lab[:, 0] + 16) / 116
    fx = fy + lab[:, 1] / 500
    fz = fy - lab[:, 2] / 200
    transformed = np.column_stack((fx, fy, fz))
    epsilon = 216 / 24389
    kappa = 24389 / 27
    xyz_scaled = np.where(
        transformed**3 > epsilon,
        transformed**3,
        (116 * transformed - 16) / kappa,
    )
    xyz = xyz_scaled * np.asarray([0.95047, 1.0, 1.08883])
    xyz_to_rgb = np.asarray(
        [
            [3.2404542, -0.9692660, 0.0556434],
            [-1.5371385, 1.8760108, -0.2040259],
            [-0.4985314, 0.0415560, 1.0572252],
        ]
    )
    linear = xyz @ xyz_to_rgb
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.maximum(linear, 0) ** (1 / 2.4) - 0.055,
    )
    return np.round(np.clip(srgb, 0, 1) * 255).astype(np.uint8)


def rgb_to_hex(rgb: np.ndarray) -> str:
    """Format an RGB triplet as uppercase hexadecimal."""
    return "#" + "".join(f"{int(channel):02X}" for channel in rgb)


def extract_dominant_color(rgb_pixels: np.ndarray) -> dict:
    """Extract the largest perceptual K-means color cluster."""
    if len(rgb_pixels) < N_CANDIDATES_PER_REGION:
        raise ValueError("Region has fewer pixels than color candidates")
    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(MAX_SAMPLES, len(rgb_pixels))
    sample_indices = rng.choice(len(rgb_pixels), size=sample_size, replace=False)
    sample_lab = srgb_to_lab(rgb_pixels[sample_indices])
    model = KMeans(
        n_clusters=N_CANDIDATES_PER_REGION,
        n_init=10,
        random_state=RANDOM_STATE,
    )
    model.fit(sample_lab)
    labels = model.predict(srgb_to_lab(rgb_pixels))
    counts = np.bincount(labels, minlength=N_CANDIDATES_PER_REGION)
    dominant_cluster = int(np.argmax(counts))
    rgb = lab_to_srgb(model.cluster_centers_[dominant_cluster])[0]
    return {
        "hex": rgb_to_hex(rgb),
        "rgb": [int(value) for value in rgb],
        "dominance": float(counts[dominant_cluster] / counts.sum()),
    }


def stage_04_extract_colors() -> None:
    """Extract and save dominant colors for anatomical regions."""
    LOGGER.info("Stage 04/04: extracting regional dominant colors")
    texture_path = OUTPUT_DIR / "texture.png"
    colors_path = OUTPUT_DIR / "colors.json"
    require_file(texture_path, "Texture")
    net_data = {
        pose_name: load_json(RAW_DATA_DIR / f"dove-{pose_name}-net.json")
        for pose_name in POSE_NAMES
    }
    with Image.open(texture_path) as source:
        texture_image = source.convert("RGBA")
    texture_rgba = np.asarray(texture_image, dtype=np.uint8)
    texture_height, texture_width = texture_rgba.shape[:2]
    if texture_height != texture_width:
        raise ValueError("The y=x symmetric texture must be square")
    foreground_mask = texture_rgba[..., 3] >= ALPHA_THRESHOLD

    colors = []
    for region_name, face_names in REGION_FACES.items():
        mask_image = Image.new("L", (texture_width, texture_height), 0)
        draw = ImageDraw.Draw(mask_image)
        found_faces = set()
        for data in net_data.values():
            points = collect_points(data)
            net_width, net_height = data["resolution"]
            scale_x = texture_width / net_width
            scale_y = texture_height / net_height
            for face_name in face_names:
                if face_name not in data["faces"]:
                    continue
                polygon = [
                    (points[name][0] * scale_x, points[name][1] * scale_y)
                    for name in data["faces"][face_name]
                ]
                draw.polygon(polygon, fill=255)
                found_faces.add(face_name)
        if not found_faces:
            raise KeyError(f"No net faces found for region: {region_name}")
        region_mask = np.asarray(mask_image) > 0
        region_mask = (region_mask | region_mask.T) & foreground_mask
        if not region_mask.any():
            raise ValueError(f"Region has no foreground pixels: {region_name}")
        color = extract_dominant_color(texture_rgba[..., :3][region_mask])
        colors.append({"region": region_name, **color})
        LOGGER.info(
            "%s: %s RGB%s dominance=%.1f%%",
            region_name,
            color["hex"],
            tuple(color["rgb"]),
            color["dominance"] * 100,
        )

    result = {
        "colors": [
            {key: color[key] for key in ("region", "hex", "rgb")}
            for color in colors
        ]
    }
    colors_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOGGER.info("Saved regional color data to: %s", colors_path)


def run_pipeline() -> None:
    """Run all four integrated stages with memory cleanup between them."""
    ensure_project_dirs()
    reset_runtime_memory("start")
    try:
        stage_01_generate_images()
        reset_runtime_memory("after stage 01")
        stage_02_warp_patterns()
        reset_runtime_memory("after stage 02")
        stage_03_map_texture()
        reset_runtime_memory("after stage 03")
        stage_04_extract_colors()
        LOGGER.info("Origami texturing pipeline completed successfully")
    finally:
        reset_runtime_memory("end")


def main() -> None:
    """Configure logging and run the complete pipeline."""
    configure_logging()
    run_pipeline()


if __name__ == "__main__":
    main()
