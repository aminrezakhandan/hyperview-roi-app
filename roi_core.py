"""Core numerical and contour utilities for the Multimodal ROI Analyzer."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
from PIL import Image
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


def canvas_size(width: int, height: int, max_width: int = 900, max_height: int = 1200) -> tuple[int, int]:
    """Display size for the drawing canvas: scaled to fit max_width x max_height.

    Small pixel maps (e.g. 80 x 60 CSV grids) are scaled *up* so they can be
    drawn on; large images are scaled down. Aspect ratio is preserved.
    """
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive.")
    scale = min(max_width / width, max_height / height)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def trim_nan_border(array: np.ndarray) -> np.ndarray:
    """Drop leading/trailing rows and columns that contain no finite value.

    Camera CSV exports often start with a text header row (``File: ...``) and a
    label column (``Frame 1``); after numeric coercion these are all-NaN and would
    otherwise shift every pixel (and any contour) by one row/column.
    """
    data = np.asarray(array, dtype=float)
    if data.ndim != 2:
        raise ValueError("trim_nan_border expects a 2-D array.")
    finite = np.isfinite(data)
    rows = np.flatnonzero(finite.any(axis=1))
    cols = np.flatnonzero(finite.any(axis=0))
    if rows.size == 0:
        return data
    return data[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


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
    is_float = arr.dtype.kind == "f"
    arr = arr[..., :3].astype(float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros((*arr.shape[:2], 3), dtype=np.uint8)
    low = np.nanmin(arr[finite])
    high = np.nanmax(arr[finite])
    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if is_float and low >= 0 and high <= 1:
        # Float images in 0..1 (checked before 0..255 so they are not truncated to 0/1).
        out = clean * 255
    elif low >= 0 and high <= 255:
        out = clean
    else:
        p1, p99 = np.nanpercentile(arr[finite], [1, 99])
        if p99 <= p1:
            p99 = p1 + 1.0
        out = np.clip((clean - p1) / (p99 - p1), 0, 1) * 255
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def flatten_alpha(image: Image.Image, background: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Return an RGB uint8 array, compositing any transparency onto `background`.

    Plain ``convert("RGB")`` discards alpha and exposes the hidden colour of
    transparent pixels (often black), which several colormaps would then decode
    as a real low value. White is rejected by the default mismatch
    threshold for every supported colormap except ``hot``, whose top end is white.
    """
    has_alpha = image.mode in {"RGBA", "LA", "PA", "La", "RGBa"} or "transparency" in image.info
    if has_alpha:
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (*background, 255))
        canvas.alpha_composite(rgba)
        return np.asarray(canvas.convert("RGB"), dtype=np.uint8)
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


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
    if cmap_name not in colormaps:
        raise ValueError(f"Unknown colormap: {cmap_name!r}")
    if not (np.isfinite(value_min) and np.isfinite(value_max)):
        raise ValueError("value_min and value_max must be finite numbers.")
    if lut_size < 2:
        raise ValueError("lut_size must be at least 2.")
    source = ensure_rgb_uint8(rgb)
    if source.size == 0:
        raise ValueError("Cannot decode an empty image.")
    positions = np.linspace(0.0, 1.0, lut_size)
    if reverse:
        positions = positions[::-1]
    lut = np.asarray(colormaps[cmap_name](positions, bytes=True)[..., :3], dtype=float)
    tree = cKDTree(lut)
    # Query each distinct colour once: screenshots contain far fewer unique
    # colours than pixels, which keeps memory/time low for large images.
    packed = (
        (source[..., 0].astype(np.uint32) << 16)
        | (source[..., 1].astype(np.uint32) << 8)
        | source[..., 2].astype(np.uint32)
    )
    unique_colors, inverse = np.unique(packed.ravel(), return_inverse=True)
    unique_rgb = np.stack(
        [(unique_colors >> 16) & 255, (unique_colors >> 8) & 255, unique_colors & 255], axis=1
    ).astype(float)
    unique_dist, unique_idx = tree.query(unique_rgb)
    inverse = inverse.reshape(source.shape[:2])
    distance_map = unique_dist[inverse]
    normalized = unique_idx[inverse] / float(lut_size - 1)
    values = value_min + normalized * (value_max - value_min)
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


def _flatten_path(commands: Iterable[Any], steps: int = 12) -> list[tuple[float, float]]:
    """Flatten Fabric.js path commands (M, L, Q, C, Z) into polyline vertices.

    Freehand strokes are quadratic Beziers; using only their control points as
    vertices (as before) misplaces the boundary, so curves are sampled instead.
    """
    points: list[tuple[float, float]] = []
    current: tuple[float, float] | None = None
    t = np.linspace(0.0, 1.0, steps + 1)[1:]
    for command in commands:
        op = str(command[0]).upper()
        nums = [float(v) for v in command[1:] if isinstance(v, (int, float))]
        if op in ("M", "L") and len(nums) >= 2:
            current = (nums[-2], nums[-1])
            points.append(current)
        elif op == "Q" and len(nums) >= 4 and current is not None:
            (x0, y0), (cx, cy), (x1, y1) = current, (nums[0], nums[1]), (nums[2], nums[3])
            bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t**2 * x1
            by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t**2 * y1
            points.extend(zip(bx.tolist(), by.tolist()))
            current = (x1, y1)
        elif op == "C" and len(nums) >= 6 and current is not None:
            (x0, y0) = current
            bx = (1-t)**3*x0 + 3*(1-t)**2*t*nums[0] + 3*(1-t)*t**2*nums[2] + t**3*nums[4]
            by = (1-t)**3*y0 + 3*(1-t)**2*t*nums[1] + 3*(1-t)*t**2*nums[3] + t**3*nums[5]
            points.extend(zip(bx.tolist(), by.tolist()))
            current = (nums[4], nums[5])
    return points


def _object_center(obj: dict[str, Any], width: float, height: float) -> tuple[float, float]:
    """Centre of a Fabric.js object in canvas pixels.

    `left`/`top` give the position of the point named by originX/originY *in the
    rotated object* (the circle tool, for example, sends the drag start point as
    left/center with the drag direction as `angle`). The centre is therefore that
    point plus the rotated offset to the box centre; rotating the unrotated box
    about its own corner-derived centre (the earlier approach) misplaces any
    rotated object whose origin is not the centre.
    """
    left = float(obj.get("left", 0.0))
    top = float(obj.get("top", 0.0))
    offset_x = (0.5 - _origin_fraction(obj.get("originX", "left"), "x")) * width
    offset_y = (0.5 - _origin_fraction(obj.get("originY", "top"), "y")) * height
    theta = math.radians(float(obj.get("angle", 0.0)))
    cosine, sine = math.cos(theta), math.sin(theta)
    return (
        left + cosine * offset_x - sine * offset_y,
        top + sine * offset_x + cosine * offset_y,
    )


def _polygon_points_from_object(obj: dict[str, Any]) -> list[tuple[float, float]]:
    obj_type = str(obj.get("type", "")).lower()
    scale_x = float(obj.get("scaleX", 1.0))
    scale_y = float(obj.get("scaleY", 1.0))
    angle = float(obj.get("angle", 0.0))
    width = float(obj.get("width", 0.0)) * scale_x
    height = float(obj.get("height", 0.0)) * scale_y
    center_x, center_y = _object_center(obj, width, height)
    corner_x, corner_y = center_x - width / 2.0, center_y - height / 2.0  # unrotated box corner

    if obj_type == "polygon":
        raw = [(float(p["x"]), float(p["y"])) for p in obj.get("points", [])]
        if len(raw) < 3:
            return []
        min_x = min(x for x, _ in raw)
        min_y = min(y for _, y in raw)
        points = [(corner_x + (x - min_x) * scale_x, corner_y + (y - min_y) * scale_y) for x, y in raw]
    elif obj_type == "path":
        raw = _flatten_path(obj.get("path", []))
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

    return _rotate_points(points, angle, (center_x, center_y))


def object_to_polygon(obj: dict[str, Any]) -> list[tuple[float, float]]:
    """Return a supported Fabric.js object as polygon vertices in canvas pixels.

    Circles/ellipses are approximated by a 1440-vertex polygon (area error
    < 1e-5). Unsupported or degenerate objects return an empty list.
    """
    obj_type = str(obj.get("type", "")).lower()
    if obj_type in {"polygon", "path", "rect"}:
        points = _polygon_points_from_object(obj)
        return points if len(points) >= 3 else []
    if obj_type in {"circle", "ellipse"}:
        scale_x = float(obj.get("scaleX", 1.0))
        scale_y = float(obj.get("scaleY", 1.0))
        width = float(obj.get("width", obj.get("radius", 0.0) * 2.0)) * scale_x
        height = float(obj.get("height", obj.get("radius", 0.0) * 2.0)) * scale_y
        if width <= 0 or height <= 0:
            return []
        cx, cy = _object_center(obj, width, height)
        theta = np.linspace(0.0, 2.0 * np.pi, 1440, endpoint=False)
        ring = list(zip((cx + width / 2.0 * np.cos(theta)).tolist(), (cy + height / 2.0 * np.sin(theta)).tolist()))
        return _rotate_points(ring, float(obj.get("angle", 0.0)), (cx, cy))
    return []


def rasterize_polygon(
    points: Iterable[tuple[float, float]],
    height: int,
    width: int,
    *,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    chunk_pixels: int = 2_000_000,
) -> np.ndarray:
    """Boolean mask of pixels whose *centre* lies inside the polygon.

    Vertices are multiplied by x_scale/y_scale (canvas -> source pixels) and the
    polygon is sampled directly on the source grid, so the mask area is an
    unbiased estimate of the drawn area at source resolution (pixel (i, j) covers
    [j, j+1) x [i, i+1), centre at (j+0.5, i+0.5)).
    """
    from matplotlib.path import Path as MplPath

    scaled = np.asarray([(x * x_scale, y * y_scale) for x, y in points], dtype=float)
    mask = np.zeros((height, width), dtype=bool)
    if len(scaled) < 3 or not np.isfinite(scaled).all():
        return mask
    x0 = max(int(np.floor(scaled[:, 0].min() - 0.5)), 0)
    x1 = min(int(np.ceil(scaled[:, 0].max() - 0.5)) + 1, width)
    y0 = max(int(np.floor(scaled[:, 1].min() - 0.5)), 0)
    y1 = min(int(np.ceil(scaled[:, 1].max() - 0.5)) + 1, height)
    if x1 <= x0 or y1 <= y0:
        return mask
    path = MplPath(scaled)
    xs = np.arange(x0, x1) + 0.5
    rows_per_chunk = max(1, chunk_pixels // (x1 - x0))
    for start in range(y0, y1, rows_per_chunk):
        stop = min(start + rows_per_chunk, y1)
        yy, xx = np.meshgrid(np.arange(start, stop) + 0.5, xs, indexing="ij")
        inside = path.contains_points(np.column_stack([xx.ravel(), yy.ravel()]))
        mask[start:stop, x0:x1] = inside.reshape(stop - start, x1 - x0)
    return mask


def masks_from_canvas(
    json_data: dict[str, Any] | None,
    *,
    canvas_width: int,
    canvas_height: int,
    original_width: int,
    original_height: int,
) -> list[np.ndarray]:
    """Convert all supported drawn objects into masks at source-image resolution.

    Contours are rasterized directly on the source grid (not on the downscaled
    canvas and then resized), so reported pixel counts/areas reflect the drawn
    shape rather than canvas-resolution rounding.
    """
    if not json_data or canvas_width <= 0 or canvas_height <= 0:
        return []
    x_scale = original_width / float(canvas_width)
    y_scale = original_height / float(canvas_height)
    masks = []
    for obj in json_data.get("objects", []):
        polygon = object_to_polygon(obj)
        if not polygon:
            continue
        mask = rasterize_polygon(
            polygon, original_height, original_width, x_scale=x_scale, y_scale=y_scale
        )
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



# ---------------------------------------------------------------------------
# Automatic foot regions (plantar thermal images) - exploratory
# ---------------------------------------------------------------------------

FOOT_REGION_NAMES = ("Hindfoot", "Midfoot", "Forefoot")
HEEL_SIDES = ("bottom", "top", "left", "right")
_HEEL_DIRECTION = {"bottom": (0.0, 1.0), "top": (0.0, -1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold over the finite values (maximises between-class variance)."""
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("No finite values to threshold.")
    low, high = float(data.min()), float(data.max())
    if high <= low:
        raise ValueError("Image has a single value; cannot separate foot from background.")
    hist, edges = np.histogram(data, bins=bins, range=(low, high))
    weights = hist.astype(float)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(weights)
    w1 = w0[-1] - w0
    sum0 = np.cumsum(weights * centers)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean0 = sum0 / w0
        mean1 = (sum0[-1] - sum0) / w1
        between = w0 * w1 * (mean0 - mean1) ** 2
    between[~np.isfinite(between)] = -1.0
    # Take the middle of the plateau of tied maxima (not the first bin), so a
    # two-level image gets a threshold midway between the levels.
    best = np.flatnonzero(np.isclose(between, between.max(), rtol=1e-9))
    return float(centers[int(best[0] + best[-1]) // 2])


def strict_threshold(values: np.ndarray) -> tuple[float, float]:
    """Return (base, core) thresholds: Otsu, and the midpoint between it and a
    second Otsu level computed on the pixels above it.

    The core threshold sits above warm intermediate material (bedding, clothing)
    that the base threshold lumps in with the foot. Falls back to the base value
    when the upper class cannot be split further.
    """
    base = otsu_threshold(values)
    data = np.asarray(values, dtype=float)
    upper = data[np.isfinite(data) & (data > base)]
    try:
        second = otsu_threshold(upper)
    except ValueError:
        return base, base
    return base, (base + second) / 2.0 if second > base else base


def default_morphology_radii(shape: tuple[int, int]) -> tuple[int, int]:
    """(opening_radius, grow_radius) in pixels, scaled to the image size."""
    short_side = min(shape)
    return max(1, int(round(short_side * 0.0125))), max(1, int(round(short_side * 0.017)))


def _disk(radius: int) -> np.ndarray:
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return (xx**2 + yy**2) <= radius**2


def segment_feet(
    values: np.ndarray,
    *,
    threshold: float | None = None,
    core_threshold: float | None = None,
    n_feet: int = 1,
    min_area_fraction: float = 0.005,
    opening_radius: int = 0,
    grow_radius: int = 0,
) -> tuple[list[np.ndarray], float]:
    """Segment the warmest connected blob(s): returns (masks left-to-right, threshold used).

    Default: pixels with value > threshold (Otsu if None; NaN excluded).
    Strict mode (`core_threshold` given): seed from pixels above the higher
    `core_threshold`, optionally cut thin bridges with a morphological opening of
    `opening_radius` px, then grow the seed back out by `grow_radius` px but only
    over pixels above `threshold`. This keeps edges/heel/toes while dropping
    warm material that merely touches the foot.
    Holes are filled and the `n_feet` largest components larger than
    `min_area_fraction` of the image are kept.
    """
    from scipy.ndimage import binary_dilation, binary_fill_holes, binary_opening, label

    data = np.asarray(values, dtype=float)
    if data.ndim != 2:
        raise ValueError("segment_feet expects a 2-D measurement map.")
    if n_feet < 1:
        raise ValueError("n_feet must be at least 1.")
    cut = otsu_threshold(data) if threshold is None else float(threshold)
    finite = np.isfinite(data)
    candidate = finite & (data > cut)
    if core_threshold is not None:
        core = finite & (data > float(core_threshold))
        if opening_radius > 0:
            core = binary_opening(core, structure=_disk(int(opening_radius)))
        if grow_radius > 0:
            candidate = binary_dilation(core, structure=_disk(int(grow_radius))) & candidate
        else:
            candidate = core
    elif opening_radius > 0:
        candidate = binary_opening(candidate, structure=_disk(int(opening_radius)))
    used = cut if core_threshold is None else float(core_threshold)
    labels, count = label(candidate)
    if count == 0:
        return [], used
    sizes = np.bincount(labels.ravel())[1:]
    order = [i for i in np.argsort(sizes)[::-1] if sizes[i] >= min_area_fraction * data.size][:n_feet]
    masks = [binary_fill_holes(labels == (i + 1)) for i in order]
    masks.sort(key=lambda m: float(np.nonzero(m)[1].mean()))
    return masks, used


def split_foot_regions(
    mask: np.ndarray,
    heel_side: str,
    fractions: tuple[float, float] = (1 / 3, 2 / 3),
) -> dict[str, np.ndarray]:
    """Split one foot mask into hindfoot / midfoot / forefoot along its long axis.

    The long axis is the first principal axis of the mask pixels. Position t runs
    from 0 at the heel end to 1 at the toe end (heel end = the end towards
    `heel_side`), and the regions are t < f1, f1 <= t < f2, t >= f2 of the
    projected foot length. The three regions are disjoint and their union is
    exactly the input mask.
    """
    if heel_side not in _HEEL_DIRECTION:
        raise ValueError(f"heel_side must be one of {HEEL_SIDES}")
    f1, f2 = fractions
    if not (0.0 < f1 < f2 < 1.0):
        raise ValueError("fractions must satisfy 0 < hindfoot/midfoot < midfoot/forefoot < 1.")
    roi = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(roi)
    if xs.size < 3:
        raise ValueError("Foot mask is too small to split.")
    coords = np.column_stack([xs + 0.5, ys + 0.5])
    centered = coords - coords.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if float(np.dot(axis, _HEEL_DIRECTION[heel_side])) < 0:
        axis = -axis  # axis now points towards the heel
    projection = centered @ axis
    span = float(projection.max() - projection.min())
    if span <= 0:
        raise ValueError("Foot mask has no length along its long axis.")
    t = (projection.max() - projection) / span  # 0 at heel, 1 at toes
    bins = np.digitize(t, [f1, f2])  # 0,1,2
    regions = {}
    for index, name in enumerate(FOOT_REGION_NAMES):
        region = np.zeros(roi.shape, dtype=bool)
        region[ys[bins == index], xs[bins == index]] = True
        regions[name] = region
    return regions


def _simplify_polyline(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification of a closed ring (iterative)."""
    if len(points) < 4:
        return points
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        a, b = points[start], points[end]
        segment = b - a
        norm = float(np.hypot(*segment))
        inner = points[start + 1 : end] - a
        if norm == 0:
            distances = np.hypot(inner[:, 0], inner[:, 1])
        else:
            distances = np.abs(segment[0] * inner[:, 1] - segment[1] * inner[:, 0]) / norm
        far = int(np.argmax(distances))
        if distances[far] > tolerance:
            split = start + 1 + far
            keep[split] = True
            stack.extend([(start, split), (split, end)])
    return points[keep]


def mask_to_polygon(mask: np.ndarray, tolerance: float = 0.6) -> list[tuple[float, float]]:
    """Outer boundary of the largest component as polygon vertices in pixel-edge
    coordinates (pixel (i, j) covers [j, j+1) x [i, i+1)); holes are ignored."""
    import contourpy

    roi = np.pad(np.asarray(mask, dtype=float), 1)
    lines = contourpy.contour_generator(z=roi).lines(0.5)
    if not lines:
        return []
    ring = max(lines, key=len)
    ring = np.asarray(ring, dtype=float) - 0.5  # padded index -> source pixel-edge coordinates
    if np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    ring = _simplify_polyline(np.vstack([ring, ring[:1]]), tolerance)[:-1]
    return [(float(x), float(y)) for x, y in ring]


def masks_to_fabric_drawing(
    masks: list[np.ndarray],
    *,
    canvas_width: int,
    canvas_height: int,
    original_width: int,
    original_height: int,
    colors: tuple[str, ...] = ("#00FFFF", "#FFD200", "#FF5050"),
) -> dict[str, Any]:
    """Build a Fabric.js drawing (editable polygons in canvas pixels) from masks."""
    x_scale = canvas_width / float(original_width)
    y_scale = canvas_height / float(original_height)
    objects = []
    for index, mask in enumerate(masks):
        polygon = mask_to_polygon(mask)
        if len(polygon) < 3:
            continue
        pts = np.asarray([(x * x_scale, y * y_scale) for x, y in polygon])
        low = pts.min(axis=0)
        high = pts.max(axis=0)
        color = colors[index % len(colors)]
        objects.append(
            {
                "type": "polygon",
                "version": "4.4.0",
                "originX": "left",
                "originY": "top",
                "left": float(low[0]),
                "top": float(low[1]),
                "width": float(high[0] - low[0]),
                "height": float(high[1] - low[1]),
                "fill": "rgba(255, 30, 30, 0.22)",
                "stroke": color,
                "strokeWidth": 3,
                "scaleX": 1,
                "scaleY": 1,
                "angle": 0,
                "points": [{"x": float(x - low[0]), "y": float(y - low[1])} for x, y in pts],
            }
        )
    return {"version": "4.4.0", "objects": objects}


# ---------------------------------------------------------------------------
# Shape-based foot template fit (plantar view) - exploratory
# ---------------------------------------------------------------------------
#
# A generic plantar foot outline is fitted to the image by position, length,
# rotation, width and mirroring, scoring edge alignment and inside/outside
# contrast rather than a temperature cut-off. The outline below is an
# approximate, hand-designed generic shape (NOT anthropometric data): feet with
# flat/high arches, hallux valgus, amputations etc. will deviate and need manual
# correction.

# (t, medial half-width, lateral half-width) as fractions of foot length;
# t = 0 at the heel, 1 at the tip of the longest (big) toe.
_TEMPLATE_PROFILE = np.array(
    [
        (0.000, 0.000, 0.000),
        (0.015, 0.060, 0.065),
        (0.050, 0.100, 0.105),
        (0.120, 0.125, 0.130),
        (0.220, 0.125, 0.135),
        (0.320, 0.105, 0.140),
        (0.420, 0.085, 0.145),
        (0.520, 0.100, 0.150),
        (0.620, 0.150, 0.165),
        (0.720, 0.205, 0.175),
        (0.800, 0.205, 0.160),
        (0.880, 0.190, 0.120),
        (0.950, 0.130, 0.060),
        (0.985, 0.070, 0.000),
        (1.000, 0.000, 0.000),
    ]
)


def _template_samples(n: int = 120, arch: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Template half-widths at n positions. `arch` scales the medial edge around
    the arch (t ~ 0.42): 1 = generic, > 1 = flatter/fuller arch, < 1 = higher arch."""
    from scipy.interpolate import PchipInterpolator

    t = np.linspace(0.0, 1.0, n)
    medial = PchipInterpolator(_TEMPLATE_PROFILE[:, 0], _TEMPLATE_PROFILE[:, 1])(t)
    medial = medial * (1.0 + (float(arch) - 1.0) * np.exp(-(((t - 0.42) / 0.16) ** 2)))
    lateral = PchipInterpolator(_TEMPLATE_PROFILE[:, 0], _TEMPLATE_PROFILE[:, 2])(t)
    return t, medial, lateral


def template_landmarks() -> tuple[float, float]:
    """Region boundaries (as fractions of foot length from the heel) derived from
    the template's shape: the hindfoot ends where the medial edge starts to curve
    into the arch (medial half-width falls to 90 % of its heel maximum), and the
    midfoot ends at the ball of the foot (widest point of the forefoot half)."""
    t, medial, lateral = _template_samples(400)
    heel = t < 0.3
    peak_index = int(np.argmax(np.where(heel, medial, -1.0)))
    after = np.flatnonzero((t > t[peak_index]) & (medial <= 0.9 * medial[peak_index]))
    arch_start = float(t[after[0]])
    half = t >= 0.5
    ball = float(t[half][int(np.argmax((medial + lateral)[half]))])
    return arch_start, ball


def _foot_frame(params: dict[str, Any], heel_side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (heel_point, toe_direction, medial_direction) in image coordinates."""
    toe0 = -np.asarray(_HEEL_DIRECTION[heel_side], dtype=float)
    theta = math.radians(float(params["theta"]))
    c, s = math.cos(theta), math.sin(theta)
    toe = np.array([c * toe0[0] - s * toe0[1], s * toe0[0] + c * toe0[1]])
    side = np.array([-toe[1], toe[0]]) * (1.0 if params.get("mirror", 1) >= 0 else -1.0)
    centre = np.array([float(params["cx"]), float(params["cy"])])
    heel_point = centre - 0.5 * float(params["length"]) * toe
    return heel_point, toe, side


def template_polygon(params: dict[str, Any], heel_side: str, n: int = 120) -> np.ndarray:
    """Fitted template outline as (N, 2) vertices in pixel-edge coordinates."""
    t, medial, lateral = _template_samples(n, float(params.get("arch", 1.0)))
    heel_point, toe, side = _foot_frame(params, heel_side)
    length = float(params["length"])
    aspect = float(params.get("aspect", 1.0))
    axis = heel_point[None, :] + length * t[:, None] * toe[None, :]
    left = axis + (length * aspect * medial)[:, None] * side[None, :]
    right = axis - (length * aspect * lateral)[:, None] * side[None, :]
    return np.vstack([left, right[::-1][1:-1]])


def template_regions(
    params: dict[str, Any],
    heel_side: str,
    shape: tuple[int, int],
    boundaries: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Hindfoot / midfoot / forefoot masks of the fitted template (disjoint; their
    union is the template outline). Boundaries default to the template landmarks."""
    h, w = shape
    f1, f2 = boundaries if boundaries is not None else template_landmarks()
    outline = rasterize_polygon(template_polygon(params, heel_side), h, w)
    heel_point, toe, _ = _foot_frame(params, heel_side)
    ys, xs = np.nonzero(outline)
    position = ((np.column_stack([xs + 0.5, ys + 0.5]) - heel_point) @ toe) / float(params["length"])
    bins = np.digitize(position, [f1, f2])
    regions = {}
    for index, name in enumerate(FOOT_REGION_NAMES):
        mask = np.zeros(shape, dtype=bool)
        mask[ys[bins == index], xs[bins == index]] = True
        regions[name] = mask
    return regions


def _sampling_geometry(polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Outline points and outward unit normals for a closed polygon."""
    nxt = np.roll(polygon, -1, axis=0)
    prev = np.roll(polygon, 1, axis=0)
    tangent = nxt - prev
    normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    norm = np.hypot(normal[:, 0], normal[:, 1])
    norm[norm == 0] = 1.0
    normal = normal / norm[:, None]
    signed_area = 0.5 * np.sum(polygon[:, 0] * nxt[:, 1] - nxt[:, 0] * polygon[:, 1])
    if signed_area < 0:
        normal = -normal
    return polygon, normal


def fit_foot_template(
    values: np.ndarray,
    heel_side: str,
    seed_mask: np.ndarray,
    *,
    sigma: float = 1.5,
) -> dict[str, Any]:
    """Fit the generic foot outline to one foot; returns params, score, polygon.

    `seed_mask` only provides the starting position/size (a rough blob); the fit
    then maximises (a) how strongly the temperature falls off across the outline
    (edge alignment) and (b) inside-minus-surrounding contrast, over centre,
    length, rotation (+/-45 degrees), width (0.8-1.3x), arch fullness (0.8-2.2x) and
    left/right mirroring.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates
    from scipy.optimize import minimize

    if heel_side not in _HEEL_DIRECTION:
        raise ValueError(f"heel_side must be one of {HEEL_SIDES}")
    data = np.asarray(values, dtype=float)
    seed = np.asarray(seed_mask, dtype=bool)
    if data.ndim != 2 or seed.shape != data.shape or seed.sum() < 20:
        raise ValueError("Need a 2-D map and a seed blob of at least 20 pixels.")
    finite = np.isfinite(data)
    low, high = np.percentile(data[finite], [1, 99])
    scale = max(high - low, 1e-9)
    norm = np.clip((np.where(finite, data, low) - low) / scale, 0.0, 1.0)
    smooth = gaussian_filter(norm, sigma)
    gy, gx = np.gradient(smooth)
    gscale = max(float(np.percentile(np.hypot(gx, gy), 95)), 1e-9)

    def sample(field: np.ndarray, points: np.ndarray) -> np.ndarray:
        # pixel-edge coordinates -> index coordinates (centre of pixel i is i + 0.5)
        return map_coordinates(field, [points[:, 1] - 0.5, points[:, 0] - 0.5], order=1, mode="nearest")

    def score(vector: np.ndarray, mirror: int) -> float:
        params = {"cx": vector[0], "cy": vector[1], "length": vector[2], "theta": vector[3],
                  "aspect": vector[4], "arch": vector[5], "mirror": mirror}
        polygon = template_polygon(params, heel_side, n=80)
        points, normals = _sampling_geometry(polygon)
        directional = -(sample(gx, points) * normals[:, 0] + sample(gy, points) * normals[:, 1])
        edge = float(np.mean(np.clip(directional / gscale, -1.0, 1.0)))
        ring = points + normals * (0.05 * vector[2])
        centre = polygon.mean(axis=0)
        inner = centre[None, :] + 0.55 * (points - centre[None, :])
        contrast = float(np.mean(sample(smooth, inner)) - np.mean(sample(smooth, ring)))
        return (edge + 0.5 * contrast - 0.3 * abs(math.log(max(vector[4], 1e-3)))
                - 0.15 * abs(math.log(max(vector[5], 1e-3))))

    ys, xs = np.nonzero(seed)
    centre0 = np.array([xs.mean() + 0.5, ys.mean() + 0.5])
    toe0 = -np.asarray(_HEEL_DIRECTION[heel_side], dtype=float)
    extent = (np.column_stack([xs + 0.5, ys + 0.5]) - centre0) @ toe0
    length0 = float(max(extent.max() - extent.min(), 10.0))
    side0 = np.array([-toe0[1], toe0[0]])

    bounds = [
        (centre0[0] - 0.5 * length0, centre0[0] + 0.5 * length0),
        (centre0[1] - 0.5 * length0, centre0[1] + 0.5 * length0),
        (0.5 * length0, 1.6 * length0),
        (-45.0, 45.0),
        (0.8, 1.3),
        (0.8, 2.2),
    ]
    # Coarse grid over mirror/rotation/shift/length; refine only the best starts.
    # (The seed blob usually lacks cool toes/heel, so lengths above the seed's are tried.)
    candidates = []
    for mirror in (1, -1):
        for theta0 in (-24.0, -8.0, 8.0, 24.0):
            for shift in (-0.12, 0.0, 0.12):
                for stretch in (1.0, 1.12, 1.25):
                    start = np.concatenate(
                        [centre0 + shift * length0 * side0 + 0.0 * toe0, [length0 * stretch, theta0, 1.0, 1.0]]
                    )
                    start[:2] = start[:2] + (stretch - 1.0) * 0.5 * length0 * toe0  # keep the heel end fixed
                    start = np.clip(start, [b[0] for b in bounds], [b[1] for b in bounds])
                    candidates.append((score(start, mirror), mirror, start))
    candidates.sort(key=lambda item: -item[0])
    best: dict[str, Any] | None = None
    shortlist = [c for c in candidates if c[1] == 1][:4] + [c for c in candidates if c[1] == -1][:4]
    by_mirror: dict[int, dict[str, Any]] = {}
    for _, mirror, start in shortlist:
        result = minimize(lambda v: -score(v, mirror), start, method="Powell", bounds=bounds,
                          options={"xtol": 1e-2, "ftol": 1e-4, "maxiter": 40})
        if mirror not in by_mirror or -result.fun > by_mirror[mirror]["score"]:
            by_mirror[mirror] = {
                "score": float(-result.fun),
                "mirror": mirror,
                "params": {"cx": float(result.x[0]), "cy": float(result.x[1]), "length": float(result.x[2]),
                           "theta": float(result.x[3]), "aspect": float(result.x[4]), "arch": float(result.x[5]),
                           "mirror": mirror},
            }
    for fit in by_mirror.values():
        fit["polygon"] = template_polygon(fit["params"], heel_side)
    best = dict(max(by_mirror.values(), key=lambda f: f["score"]))
    best["per_mirror"] = by_mirror
    return best


def fit_feet(values: np.ndarray, heel_side: str, seed_masks: list[np.ndarray]) -> list[dict[str, Any]]:
    """Fit one template per seed. With exactly two feet in a plantar view the two
    are a left/right pair, so their mirror states are forced to be opposite and the
    assignment with the higher total score is used; otherwise each foot is free."""
    fits = [fit_foot_template(values, heel_side, seed) for seed in seed_masks]
    if len(fits) == 2:
        options = [(1, -1), (-1, 1)]
        totals = [fits[0]["per_mirror"][m0]["score"] + fits[1]["per_mirror"][m1]["score"] for m0, m1 in options]
        m0, m1 = options[int(np.argmax(totals))]
        chosen = [fits[0]["per_mirror"][m0], fits[1]["per_mirror"][m1]]
        return [{**chosen[0], "per_mirror": fits[0]["per_mirror"]}, {**chosen[1], "per_mirror": fits[1]["per_mirror"]}]
    return fits
