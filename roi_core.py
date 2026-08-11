"""Core numerical and contour utilities for the Multimodal ROI Analyzer."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw
from matplotlib import colormaps
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


SUPPORTED_COLORMAPS = (
    "turbo",
    "jet",
    "rainbow",
    "inferno",
    "magma",
    "plasma",
    "viridis",
    "hot",
    "coolwarm",
)


def normalize_to_rgb(
    values: np.ndarray,
    *,
    cmap_name: str = "gray",
    invert: bool = False,
) -> np.ndarray:
    """Convert a scalar array to an RGB display image using robust limits."""
    data = np.asarray(values, dtype=float)
    finite = np.isfinite(data)
    normalized = np.zeros(data.shape, dtype=float)
    if finite.any():
        low, high = np.nanpercentile(data[finite], [1, 99])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.nanmin(data[finite]))
            high = float(np.nanmax(data[finite]))
        if high <= low:
            high = low + 1.0
        normalized[finite] = np.clip((data[finite] - low) / (high - low), 0, 1)
    if invert:
        normalized = 1.0 - normalized
    rgba = colormaps[cmap_name](normalized, bytes=True)
    rgb = np.asarray(rgba[..., :3], dtype=np.uint8)
    rgb[~finite] = np.array([35, 35, 35], dtype=np.uint8)
    return rgb


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Return any common grayscale/RGB(A) image as uint8 RGB."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return normalize_to_rgb(arr)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"Unsupported display image shape: {arr.shape}")
    arr = arr[..., :3].astype(float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros((*arr.shape[:2], 3), dtype=np.uint8)
    low = np.nanmin(arr[finite])
    high = np.nanmax(arr[finite])
    if low >= 0 and high <= 255:
        out = np.nan_to_num(arr, nan=0.0)
    elif low >= 0 and high <= 1:
        out = np.nan_to_num(arr, nan=0.0) * 255
    else:
        p1, p99 = np.nanpercentile(arr[finite], [1, 99])
        if p99 <= p1:
            p99 = p1 + 1.0
        out = np.clip((arr - p1) / (p99 - p1), 0, 1) * 255
    return np.clip(out, 0, 255).astype(np.uint8)


def decode_false_color(
    rgb: np.ndarray,
    cmap_name: str,
    value_min: float,
    value_max: float,
    *,
    reverse: bool = False,
    max_rgb_distance: float | None = 45.0,
    lut_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate scalar values from an RGB false-color map.

    Returns (values, valid_mask, nearest_RGB_distance). This is an estimate and
    is only valid when the source uses the selected colormap and numeric range.
    """
    source = ensure_rgb_uint8(rgb).astype(float)
    positions = np.linspace(0.0, 1.0, lut_size)
    if reverse:
        positions = positions[::-1]
    lut = np.asarray(colormaps[cmap_name](positions, bytes=True)[..., :3], dtype=float)
    tree = cKDTree(lut)
    distances, indices = tree.query(source.reshape(-1, 3), workers=-1)
    normalized = indices.reshape(source.shape[:2]) / float(lut_size - 1)
    values = value_min + normalized * (value_max - value_min)
    distance_map = distances.reshape(source.shape[:2])
    if max_rgb_distance is None:
        valid = np.ones(source.shape[:2], dtype=bool)
    else:
        valid = distance_map <= float(max_rgb_distance)
    values = values.astype(float)
    values[~valid] = np.nan
    return values, valid, distance_map


def summary_statistics(
    values: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Calculate descriptive statistics inside a mask."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 2:
        raise ValueError("summary_statistics expects one 2-D measurement map.")
    selected = np.ones(data.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if selected.shape != data.shape:
        raise ValueError("Mask and data shapes do not match.")
    total = int(selected.sum())
    valid_values = data[selected & np.isfinite(data)]
    valid_n = int(valid_values.size)
    result: dict[str, float | int] = {
        "selected_pixels": total,
        "valid_pixels": valid_n,
        "missing_pixels": total - valid_n,
        "valid_fraction": valid_n / total if total else np.nan,
    }
    names = ("mean", "median", "std", "min", "p05", "p25", "p75", "p95", "max")
    if valid_n == 0:
        result.update({name: np.nan for name in names})
        return result
    q05, q25, q50, q75, q95 = np.nanpercentile(valid_values, [5, 25, 50, 75, 95])
    result.update(
        {
            "mean": float(np.nanmean(valid_values)),
            "median": float(q50),
            "std": float(np.nanstd(valid_values, ddof=1)) if valid_n > 1 else 0.0,
            "min": float(np.nanmin(valid_values)),
            "p05": float(q05),
            "p25": float(q25),
            "p75": float(q75),
            "p95": float(q95),
            "max": float(np.nanmax(valid_values)),
        }
    )
    return result


def roi_geometry(
    mask: np.ndarray,
    *,
    row_spacing: float = 1.0,
    column_spacing: float = 1.0,
) -> dict[str, float | int]:
    """Calculate area, edge-based perimeter, equivalent diameter, and bounding box."""
    roi = np.asarray(mask, dtype=bool)
    count = int(roi.sum())
    if count == 0:
        return {
            "roi_pixels": 0,
            "roi_fraction": 0.0,
            "area": 0.0,
            "perimeter": 0.0,
            "equivalent_diameter": 0.0,
            "bbox_x_min": np.nan,
            "bbox_y_min": np.nan,
            "bbox_x_max": np.nan,
            "bbox_y_max": np.nan,
        }
    padded = np.pad(roi, 1, mode="constant", constant_values=False)
    row_boundaries = np.count_nonzero(padded[1:, :] != padded[:-1, :])
    col_boundaries = np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    area = count * row_spacing * column_spacing
    perimeter = row_boundaries * column_spacing + col_boundaries * row_spacing
    ys, xs = np.nonzero(roi)
    return {
        "roi_pixels": count,
        "roi_fraction": count / roi.size,
        "area": float(area),
        "perimeter": float(perimeter),
        "equivalent_diameter": float(math.sqrt(4.0 * area / math.pi)),
        "bbox_x_min": int(xs.min()),
        "bbox_y_min": int(ys.min()),
        "bbox_x_max": int(xs.max()),
        "bbox_y_max": int(ys.max()),
    }


def _rotate_points(
    points: Iterable[tuple[float, float]],
    angle_degrees: float,
    center: tuple[float, float],
) -> list[tuple[float, float]]:
    if not angle_degrees:
        return list(points)
    theta = math.radians(angle_degrees)
    cosine, sine = math.cos(theta), math.sin(theta)
    cx, cy = center
    output = []
    for x, y in points:
        dx, dy = x - cx, y - cy
        output.append((cx + cosine * dx - sine * dy, cy + sine * dx + cosine * dy))
    return output


def _origin_fraction(value: Any, axis: str) -> float:
    """Convert a Fabric.js originX/originY value to a 0..1 fraction of width/height."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).lower()
    if axis == "x":
        return {"left": 0.0, "center": 0.5, "right": 1.0}.get(text, 0.0)
    return {"top": 0.0, "center": 0.5, "bottom": 1.0}.get(text, 0.0)


def _bbox_corner(obj: dict[str, Any], width: float, height: float) -> tuple[float, float]:
    """Fabric.js positions `left`/`top` at the point named by originX/originY
    (the freehand and polygon tools use "center", not the top-left corner)."""
    left = float(obj.get("left", 0.0))
    top = float(obj.get("top", 0.0))
    corner_x = left - _origin_fraction(obj.get("originX", "left"), "x") * width
    corner_y = top - _origin_fraction(obj.get("originY", "top"), "y") * height
    return corner_x, corner_y


def _polygon_points_from_object(obj: dict[str, Any]) -> list[tuple[float, float]]:
    obj_type = str(obj.get("type", "")).lower()
    scale_x = float(obj.get("scaleX", 1.0))
    scale_y = float(obj.get("scaleY", 1.0))
    angle = float(obj.get("angle", 0.0))
    width = float(obj.get("width", 0.0)) * scale_x
    height = float(obj.get("height", 0.0)) * scale_y
    corner_x, corner_y = _bbox_corner(obj, width, height)

    if obj_type == "polygon":
        raw = [(float(p["x"]), float(p["y"])) for p in obj.get("points", [])]
        if len(raw) < 3:
            return []
        min_x = min(x for x, _ in raw)
        min_y = min(y for _, y in raw)
        points = [(corner_x + (x - min_x) * scale_x, corner_y + (y - min_y) * scale_y) for x, y in raw]
    elif obj_type == "path":
        raw: list[tuple[float, float]] = []
        for command in obj.get("path", []):
            numbers = command[1:]
            for index in range(0, len(numbers) - 1, 2):
                if isinstance(numbers[index], (int, float)) and isinstance(numbers[index + 1], (int, float)):
                    raw.append((float(numbers[index]), float(numbers[index + 1])))
        if len(raw) < 3:
            return []
        min_x = min(x for x, _ in raw)
        min_y = min(y for _, y in raw)
        points = [(corner_x + (x - min_x) * scale_x, corner_y + (y - min_y) * scale_y) for x, y in raw]
    elif obj_type == "rect":
        points = [
            (corner_x, corner_y),
            (corner_x + width, corner_y),
            (corner_x + width, corner_y + height),
            (corner_x, corner_y + height),
        ]
    else:
        return []

    center_x = corner_x + width / 2.0
    center_y = corner_y + height / 2.0
    return _rotate_points(points, angle, (center_x, center_y))


def object_to_canvas_mask(
    obj: dict[str, Any],
    canvas_width: int,
    canvas_height: int,
) -> np.ndarray | None:
    """Rasterize one supported Fabric.js canvas object to a boolean mask."""
    obj_type = str(obj.get("type", "")).lower()
    image = Image.new("L", (canvas_width, canvas_height), 0)
    draw = ImageDraw.Draw(image)
    if obj_type in {"polygon", "path", "rect"}:
        points = _polygon_points_from_object(obj)
        if len(points) < 3:
            return None
        draw.polygon(points, fill=255)
    elif obj_type in {"circle", "ellipse"}:
        width = float(obj.get("width", obj.get("radius", 0.0) * 2.0))
        height = float(obj.get("height", obj.get("radius", 0.0) * 2.0))
        width *= float(obj.get("scaleX", 1.0))
        height *= float(obj.get("scaleY", 1.0))
        corner_x, corner_y = _bbox_corner(obj, width, height)
        draw.ellipse((corner_x, corner_y, corner_x + width, corner_y + height), fill=255)
    else:
        return None
    return np.asarray(image, dtype=np.uint8) > 0


def masks_from_canvas(
    json_data: dict[str, Any] | None,
    *,
    canvas_width: int,
    canvas_height: int,
    original_width: int,
    original_height: int,
) -> list[np.ndarray]:
    """Convert all supported drawn objects into masks at source-image resolution."""
    if not json_data:
        return []
    masks = []
    for obj in json_data.get("objects", []):
        canvas_mask = object_to_canvas_mask(obj, canvas_width, canvas_height)
        if canvas_mask is None or not canvas_mask.any():
            continue
        resized = Image.fromarray(canvas_mask.astype(np.uint8) * 255).resize(
            (original_width, original_height),
            resample=Image.Resampling.NEAREST,
        )
        mask = np.asarray(resized, dtype=np.uint8) > 0
        if mask.any():
            masks.append(mask)
    return masks


def overlay_masks(
    display_rgb: np.ndarray,
    masks: list[np.ndarray],
    *,
    alpha: float = 0.22,
) -> np.ndarray:
    """Overlay semi-transparent fills and high-contrast borders on an RGB image."""
    output = ensure_rgb_uint8(display_rgb).astype(float)
    colors = (
        (0, 255, 255),
        (255, 80, 80),
        (255, 210, 0),
        (80, 255, 100),
        (220, 100, 255),
        (80, 140, 255),
    )
    for index, mask in enumerate(masks):
        roi = np.asarray(mask, dtype=bool)
        color = np.asarray(colors[index % len(colors)], dtype=float)
        output[roi] = (1.0 - alpha) * output[roi] + alpha * color
        border = roi & ~binary_erosion(roi)
        output[border] = color
    return np.clip(output, 0, 255).astype(np.uint8)

