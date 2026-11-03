"""Streamlit app for interactive analysis of imaging and pixel-map files."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

import vendor_drawable_canvas as _drawable_canvas_module
from vendor_drawable_canvas import st_canvas

# The canvas component normally asks Streamlit's MediaFileManager for a URL to
# the background image (a path like "/media/<hash>.png"). On Streamlit
# Community Cloud the app is served behind a proxy path prefix (e.g.
# "/~/+/") that a root-relative media path doesn't include, so the request
# never reaches the app and the canvas renders with no background image.
# Embedding the image as a data URL sidesteps the MediaFileManager/proxy path
# entirely. This requires the accompanying frontend patch vendored in
# vendor_drawable_canvas/ (see its README) that stops prefixing the URL with
# the page origin, which would otherwise corrupt a data URL.
def _background_image_to_data_url(image, *_args, **_kwargs) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


_drawable_canvas_module.image_to_url = _background_image_to_data_url


def show_image(array, caption: str = "", width: int | None = None, max_display_width: int = 1600) -> None:
    """Display an image as an inline data URL.

    st.image serves files through Streamlit's media endpoint ("/media/<hash>.png"),
    which does not resolve behind Community Cloud's proxy path prefix and renders a
    broken-image icon (the same problem the canvas background works around above).
    Embedding the PNG avoids the media endpoint; nothing is written to disk.
    """
    image = Image.fromarray(ensure_rgb_uint8(array))
    if image.width > max_display_width:  # display copy only; exports keep full resolution
        image = image.resize(
            (max_display_width, max(1, round(image.height * max_display_width / image.width))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    # A data: URL string is passed through by st.image as-is (no /media/ file), so Streamlit's
    # normal sizing and captions apply.
    data_url = f"data:image/png;base64,{encoded}"
    if width:
        st.image(data_url, caption=caption or None, width=int(width))
    else:
        st.image(data_url, caption=caption or None, use_container_width=True)

from roi_core import (
    FOOT_REGION_NAMES,
    HEEL_SIDES,
    SUPPORTED_COLORMAPS,
    canvas_size,
    decode_false_color,
    default_morphology_radii,
    default_region_pairs,
    ensure_rgb_uint8,
    flatten_alpha,
    fit_feet,
    left_right_comparison,
    masks_from_canvas,
    masks_to_fabric_drawing,
    normalize_to_rgb,
    otsu_threshold,
    overlay_masks,
    roi_geometry,
    segment_feet,
    split_foot_regions,
    strict_threshold,
    template_landmarks,
    template_regions,
    trim_nan_border,
    summary_statistics,
)


st.set_page_config(page_title="Multimodal ROI Analyzer", page_icon="🩻", layout="wide")
st.title("Multimodal Imaging ROI Analyzer")
st.caption(
    "Analyze PNG/JPEG/TIFF, DICOM (.dcm), or CSV pixel maps; draw one or more contours; "
    "and export whole-image and contour-specific measurements."
)


def sanitized_stem(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return cleaned or "analysis"


def image_bytes(image: np.ndarray, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(ensure_rgb_uint8(image)).save(buffer, format=fmt)
    return buffer.getvalue()


def mask_bytes(mask: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(buffer, format="PNG")
    return buffer.getvalue()


MAX_IMAGE_PIXELS = 40_000_000  # keeps memory bounded on small hosts (e.g. Community Cloud)


def read_raster(payload: bytes) -> tuple[np.ndarray, np.ndarray, dict]:
    image = Image.open(io.BytesIO(payload))
    source_format = image.format  # exif_transpose returns a copy whose .format is None
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image is too large ({image.width} x {image.height}); "
            f"limit is {MAX_IMAGE_PIXELS:,} pixels. Crop or downscale it first."
        )
    image = ImageOps.exif_transpose(image)
    metadata = {
        "format": source_format,
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
    }
    if image.mode in {"L", "I", "I;16", "F"}:
        scalar = np.asarray(image, dtype=float)
        display = normalize_to_rgb(scalar)
        return scalar, display, metadata
    rgb = flatten_alpha(image)  # composites transparency on white instead of exposing hidden colours
    return rgb, rgb, metadata


def read_dicom(payload: bytes) -> tuple[np.ndarray, np.ndarray, dict, float | None, float | None]:
    try:
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    except ImportError as exc:
        raise RuntimeError("Install the packages in requirements.txt to read DICOM files.") from exc

    # Ensure JPEG Lossless plugins are registered before decoding.
    try:
        import pylibjpeg  # noqa: F401
        import libjpeg  # noqa: F401
    except ImportError:
        pass
    try:
        import gdcm  # noqa: F401
    except ImportError:
        pass

    dataset = pydicom.dcmread(io.BytesIO(payload), force=True)
    if not hasattr(dataset, "PixelData") and not hasattr(dataset, "FloatPixelData"):
        raise ValueError("This DICOM file has no pixel data (it may be a structured report or header-only file).")

    try:
        # pydicom 3 preferred API with explicit plugin preference
        try:
            from pydicom.pixels import pixel_array as _pixel_array

            pixels = np.asarray(
                _pixel_array(
                    dataset,
                    decoding_plugin=None,  # auto-select among available plugins
                )
            )
        except TypeError:
            pixels = np.asarray(dataset.pixel_array)
        except Exception:
            pixels = np.asarray(dataset.pixel_array)
    except Exception as exc:
        raise RuntimeError(
            "Could not decode DICOM pixel data. JPEG Lossless files need "
            "`pylibjpeg`, `pylibjpeg-libjpeg>=2.1`, and/or `python-gdcm`. "
            "From the project folder run:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install -U pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm\n"
            "Then fully stop and restart Streamlit.\n"
            f"Original error: {exc}"
        ) from exc

    frame_count = int(getattr(dataset, "NumberOfFrames", 1) or 1)

    if pixels.ndim == 2:
        selected = pixels
    elif pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
        selected = pixels[..., :3]
    elif pixels.ndim == 3:
        frame_index = st.sidebar.slider("DICOM frame", 1, int(pixels.shape[0]), 1) - 1
        selected = pixels[frame_index]
    elif pixels.ndim == 4 and pixels.shape[-1] in (3, 4):
        frame_index = st.sidebar.slider("DICOM frame", 1, int(pixels.shape[0]), 1) - 1
        selected = pixels[frame_index, ..., :3]
    else:
        raise ValueError(f"Unsupported DICOM pixel-array shape: {pixels.shape}")

    slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
    photometric = str(getattr(dataset, "PhotometricInterpretation", ""))

    if selected.ndim == 2:
        # Prefer VOI LUT / windowing when present; fall back to rescale slope/intercept.
        try:
            scalar = np.asarray(apply_voi_lut(selected, dataset), dtype=float)
            used_voi = True
        except Exception:
            scalar = selected.astype(float) * slope + intercept
            used_voi = False
        invert = photometric == "MONOCHROME1"
        display = normalize_to_rgb(scalar, invert=invert)
    else:
        scalar = selected
        display = ensure_rgb_uint8(selected)
        used_voi = False

    row_spacing = column_spacing = None
    spacing = getattr(dataset, "PixelSpacing", None)
    if spacing is not None and len(spacing) >= 2:
        row_spacing, column_spacing = float(spacing[0]), float(spacing[1])
    else:
        # Some ultrasound/secondary-capture files store spacing under ImagerPixelSpacing.
        spacing = getattr(dataset, "ImagerPixelSpacing", None)
        if spacing is not None and len(spacing) >= 2:
            row_spacing, column_spacing = float(spacing[0]), float(spacing[1])

    metadata = {
        "modality": str(getattr(dataset, "Modality", "")),
        "sop_class_uid": str(getattr(dataset, "SOPClassUID", "")),
        "transfer_syntax_uid": str(
            getattr(getattr(dataset, "file_meta", None), "TransferSyntaxUID", "")
        ),
        "rows": int(getattr(dataset, "Rows", selected.shape[0])),
        "columns": int(getattr(dataset, "Columns", selected.shape[1])),
        "frames": frame_count,
        "photometric_interpretation": photometric,
        "rescale_slope": slope,
        "rescale_intercept": intercept,
        "voi_lut_applied": used_voi,
        "pixel_spacing_mm": [row_spacing, column_spacing],
        "series_description": str(getattr(dataset, "SeriesDescription", "")),
        "study_description": str(getattr(dataset, "StudyDescription", "")),
    }
    return scalar, display, metadata, row_spacing, column_spacing


def read_compare(file) -> tuple[np.ndarray, str]:
    """Read the optional comparison file as a display RGB image (view only)."""
    payload = file.getvalue()
    if Path(file.name).suffix.lower() == ".csv":
        frame = pd.read_csv(io.BytesIO(payload), header=None)
        array = trim_nan_border(frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))
        if not np.isfinite(array).any():
            raise ValueError("No numeric pixel values were found in the comparison CSV.")
        return normalize_to_rgb(array, cmap_name="viridis"), f"CSV matrix {array.shape[1]} x {array.shape[0]}"
    _, display, meta = read_raster(payload)
    return display, f"{meta['width']} x {meta['height']} {meta['format']}"


def looks_like_dicom(payload: bytes, filename: str) -> bool:
    extension = Path(filename).suffix.lower()
    if extension in {".dcm", ".dicom", ".ima"}:
        return True
    # DICOM Part 10 files usually have "DICM" at byte offset 128.
    return len(payload) > 132 and payload[128:132] == b"DICM"


def csv_configuration(payload: bytes) -> tuple[np.ndarray, np.ndarray, dict]:
    mode = st.sidebar.radio("CSV layout", ("Matrix grid", "Long table: x, y, value"))
    if mode == "Matrix grid":
        labels = st.sidebar.checkbox("First row and first column contain labels", value=False)
        frame = pd.read_csv(io.BytesIO(payload), header=0 if labels else None)
        if labels:
            frame = frame.iloc[:, 1:]
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        array = numeric.to_numpy(dtype=float)
        if not np.isfinite(array).any():
            raise ValueError("No numeric pixel values were found in the CSV matrix.")
        array = trim_nan_border(array)  # text header row / label column (e.g. "File:", "Frame 1")
        metadata = {"csv_layout": mode, "rows": array.shape[0], "columns": array.shape[1]}
    else:
        frame = pd.read_csv(io.BytesIO(payload))
        if frame.shape[1] < 3:
            raise ValueError("Long-table CSV input needs at least x, y, and value columns.")
        columns = list(frame.columns)
        x_column = st.sidebar.selectbox("x column", columns, index=0)
        y_column = st.sidebar.selectbox("y column", columns, index=min(1, len(columns) - 1))
        value_column = st.sidebar.selectbox("Measurement column", columns, index=min(2, len(columns) - 1))
        work = frame[[x_column, y_column, value_column]].copy()
        for column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
        work = work.dropna()
        pivot = work.pivot_table(
            index=y_column,
            columns=x_column,
            values=value_column,
            aggfunc="mean",
        ).sort_index(axis=0).sort_index(axis=1)
        array = pivot.to_numpy(dtype=float)
        if not np.isfinite(array).any():
            raise ValueError("No numeric pixel values were found after pivoting the CSV.")
        metadata = {
            "csv_layout": mode,
            "x_column": str(x_column),
            "y_column": str(y_column),
            "measurement_column": str(value_column),
            "rows": array.shape[0],
            "columns": array.shape[1],
        }
    return array, normalize_to_rgb(array, cmap_name="viridis"), metadata


uploaded = st.file_uploader(
    "Upload an image, DICOM (.dcm), or pixel map",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "dcm", "dicom", "ima", "csv"],
    help="DICOM is supported: upload .dcm / .dicom / .ima, or launch with -- --image path\\to\\file.dcm",
)

# Optional local preload: streamlit run app.py -- --image "C:\path\to\file.jpg"
# or set HYPERVIEW_ROI_IMAGE to a file path.
def _cli_image_path() -> Path | None:
    import argparse
    import os

    env_path = os.environ.get("HYPERVIEW_ROI_IMAGE", "").strip()
    if env_path:
        path = Path(env_path)
        if path.is_file():
            return path
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--image", type=str, default=None)
    args, _ = parser.parse_known_args()
    if args.image:
        path = Path(args.image)
        if path.is_file():
            return path
        st.sidebar.error(f"Local image not found: {path}")
    return None


class _UploadedFile:
    """Minimal stand-in for st.UploadedFile when preloading a local path."""

    def __init__(self, path: Path):
        self.name = path.name
        self._bytes = path.read_bytes()

    def getvalue(self) -> bytes:
        return self._bytes


if uploaded is None:
    local_image = _cli_image_path()
    if local_image is not None:
        uploaded = _UploadedFile(local_image)
        st.success(f"Loaded local image: `{local_image}`")
    else:
        st.info("Upload one file to begin, or launch with `-- --image path\\to\\file.jpg`.")
        st.stop()

payload = uploaded.getvalue()
extension = Path(uploaded.name).suffix.lower()
file_hash = hashlib.sha256(payload).hexdigest()[:12]
default_row_spacing = default_column_spacing = None
approximate_decode = False
decode_details: dict = {}

try:
    if extension == ".csv":
        source, source_display, source_metadata = csv_configuration(payload)
        source_kind = "CSV"
    elif looks_like_dicom(payload, uploaded.name):
        source, source_display, source_metadata, default_row_spacing, default_column_spacing = read_dicom(payload)
        source_kind = "DICOM"
    else:
        source, source_display, source_metadata = read_raster(payload)
        source_kind = "Raster image"
except Exception as exc:
    st.error(f"Could not read this file: {exc}")
    st.stop()

st.sidebar.header("Measurement")
measurement_name = st.sidebar.text_input("Measurement name", value="Pixel value")
unit = st.sidebar.text_input("Unit", value="")

if source.ndim == 3:
    interpretation_options = (
        "Luminance / grayscale intensity",
        "Red channel",
        "Green channel",
        "Blue channel",
        "Mean RGB intensity",
        "Decode a known false-color map",
    )
    interpretation = st.sidebar.selectbox("Interpret RGB pixels as", interpretation_options)
    source_float = source.astype(float)
    if interpretation == "Red channel":
        values = source_float[..., 0]
    elif interpretation == "Green channel":
        values = source_float[..., 1]
    elif interpretation == "Blue channel":
        values = source_float[..., 2]
    elif interpretation == "Mean RGB intensity":
        values = source_float.mean(axis=2)
    elif interpretation == "Decode a known false-color map":
        cmap_name = st.sidebar.selectbox("Source colormap", SUPPORTED_COLORMAPS)
        value_min = st.sidebar.number_input("Displayed minimum", value=0.0, format="%.6f")
        value_max = st.sidebar.number_input("Displayed maximum", value=100.0, format="%.6f")
        reverse_cmap = st.sidebar.checkbox("Reverse colormap", value=False)
        max_distance = st.sidebar.slider("Maximum RGB mismatch", 0, 150, 45)
        if value_max == value_min:
            st.error("Displayed minimum and maximum must differ.")
            st.stop()
        values, decoded_valid, distance_map = decode_false_color(
            source_display,
            cmap_name,
            value_min,
            value_max,
            reverse=reverse_cmap,
            max_rgb_distance=float(max_distance),
        )
        approximate_decode = True
        decode_details = {
            "colormap": cmap_name,
            "displayed_minimum": value_min,
            "displayed_maximum": value_max,
            "reverse_colormap": reverse_cmap,
            "maximum_rgb_mismatch": max_distance,
            "decoded_valid_fraction": float(decoded_valid.mean()),
            "median_rgb_mismatch": float(np.nanmedian(distance_map)),
        }
        st.warning(
            "False-color decoding is an estimate. It is quantitative only if the selected "
            "colormap and displayed minimum/maximum exactly match the source image."
        )
    else:
        values = (
            0.2126 * source_float[..., 0]
            + 0.7152 * source_float[..., 1]
            + 0.0722 * source_float[..., 2]
        )
else:
    values = np.asarray(source, dtype=float)

finite = values[np.isfinite(values)]
data_min = float(np.nanmin(finite)) if finite.size else 0.0
data_max = float(np.nanmax(finite)) if finite.size else 1.0

with st.sidebar.expander("Optional value remapping (min / max)", expanded=False):
    st.caption(
        f"Current pixel range: {data_min:.6g} → {data_max:.6g}. "
        "Map that range linearly onto the min/max you enter "
        "(e.g. colorbar temperature limits)."
    )
    target_min = st.number_input("Mapped minimum", value=float(data_min), format="%.6f", key="map_min")
    target_max = st.number_input("Mapped maximum", value=float(data_max), format="%.6f", key="map_max")
    apply_remap = st.checkbox("Apply min/max remapping", value=False)

if apply_remap:
    if target_max == target_min:
        st.error("Mapped minimum and maximum must differ.")
        st.stop()
    if data_max == data_min:
        values = np.where(np.isfinite(values), float(target_min), np.nan)
    else:
        values = (
            float(target_min)
            + (values - data_min) / (data_max - data_min) * (float(target_max) - float(target_min))
        )

st.sidebar.header("Spatial calibration")
use_dicom_spacing = default_row_spacing is not None and default_column_spacing is not None
if use_dicom_spacing:
    st.sidebar.caption("Pixel spacing was read from DICOM metadata.")
spatial_unit = st.sidebar.text_input("Distance unit", value="mm" if use_dicom_spacing else "pixel")
row_spacing = st.sidebar.number_input(
    f"Row spacing ({spatial_unit}/pixel)",
    min_value=0.000001,
    value=float(default_row_spacing or 1.0),
    format="%.6f",
)
column_spacing = st.sidebar.number_input(
    f"Column spacing ({spatial_unit}/pixel)",
    min_value=0.000001,
    value=float(default_column_spacing or 1.0),
    format="%.6f",
)

st.sidebar.header("Compare")
compare_file = st.sidebar.file_uploader(
    "Second image to view side by side (optional)",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "csv"],
    key="compare_upload",
    help="View only: no contours or statistics are computed for this file. CSVs must be a numeric matrix.",
)

height, width = values.shape
st.write(
    f"**Source:** {uploaded.name} · **Type:** {source_kind} · "
    f"**Measurement grid:** {width} × {height} pixels"
)

with st.expander("Source details"):
    st.json(source_metadata)

if compare_file is not None:
    st.subheader("Side-by-side comparison")
    try:
        compare_display, compare_info = read_compare(compare_file)
    except Exception as exc:
        st.error(f"Could not read the comparison file: {exc}")
    else:
        left_view, right_view = st.columns(2)
        with left_view:
            show_image(source_display, caption=f"{uploaded.name} ({width} x {height})")
        with right_view:
            show_image(compare_display, caption=f"{compare_file.name} ({compare_info})")
        st.caption(
            "View only. For CSV maps the colours are scaled separately for each image, so the same colour "
            "does not mean the same value across the two panels."
        )

st.subheader("Draw contour(s)")
st.caption(
    "Polygon is recommended: click around the region and double-click the final point. "
    "Freehand contours are automatically closed. Use Transform to select, move, or delete a contour."
)
control_left, control_right = st.columns([3, 1])
with control_left:
    drawing_mode_label = st.radio(
        "Contour tool",
        ("Polygon", "Freehand", "Rectangle", "Circle", "Transform"),
        horizontal=True,
    )
with control_right:
    if "canvas_revision" not in st.session_state:
        st.session_state.canvas_revision = 0
    if st.button("Clear all contours", use_container_width=True):
        st.session_state.canvas_revision += 1
        st.session_state.pop("auto_foot", None)
        st.rerun()

with st.expander("Automatic foot regions: hindfoot / midfoot / forefoot (plantar thermal, exploratory)"):
    st.caption(
        "Recommended: 'Fit foot template' fits a generic plantar foot outline (heel, arch, ball, toes) to each foot "
        "by shape and edge alignment, so warm bedding and cool toes matter much less. Alternatively draw ONE rough "
        "outline per foot and press 'Split each drawn contour' under the canvas, or use the older threshold-based "
        "generator (which segments the warmest connected area as the foot (Otsu or manual threshold), finds its long "
        "axis, and splits the length from the heel into regions). The result is loaded onto the canvas "
        "as ordinary editable polygons: always check and correct them. Plantar views, roughly "
        "upright, with the foot warmer than the background. Crop out legends, hands and bedding first."
    )
    auto_left, auto_right = st.columns(2)
    with auto_left:
        auto_feet = st.radio("Feet in image", (1, 2), horizontal=True)
        auto_heel = st.selectbox("Heel is at the ... of the image", HEEL_SIDES, index=0)
        threshold_mode = st.radio(
            "Foot threshold",
            ("Automatic (Otsu)", "Strict (exclude warm bedding)", "Manual"),
            horizontal=True,
            help="Strict uses a higher two-level Otsu seed, cuts thin bridges, then grows back a few "
            "pixels; use it when warm bedding/clothing touches the feet. It can trim cool toe tips.",
        )
        manual_threshold = None
        if threshold_mode == "Manual":
            try:
                suggested = otsu_threshold(values)
            except ValueError:
                suggested = float(np.nanmedian(finite)) if finite.size else 0.0
            manual_threshold = st.number_input(
                "Threshold (foot > this value)",
                value=float(suggested),
                format="%.4f",
                help="Starts at the automatic Otsu level; raise it until bedding drops out.",
            )
    with auto_right:
        hind_end = st.number_input("Hindfoot ends at (% of foot length from heel)", 5.0, 90.0, 33.0, step=1.0)
        mid_end = st.number_input("Midfoot ends at (% of foot length from heel)", 10.0, 95.0, 66.0, step=1.0)
    def auto_segment():
        if threshold_mode.startswith("Strict"):
            base_cut, core_cut = strict_threshold(values)
            open_r, grow_r = default_morphology_radii(values.shape)
            return segment_feet(
                values,
                threshold=base_cut,
                core_threshold=core_cut,
                n_feet=int(auto_feet),
                opening_radius=open_r,
                grow_radius=grow_r,
            )
        return segment_feet(values, threshold=manual_threshold, n_feet=int(auto_feet))

    if st.checkbox("Preview the automatic foot mask with these settings"):
        try:
            preview_masks, preview_threshold = auto_segment()
            show_image(
                overlay_masks(source_display, preview_masks),
                caption=f"Segmented foot area (threshold {preview_threshold:.4g}). If it includes bedding or "
                "misses toes, use the drawn-outline method below the canvas instead.",
                width=480,
            )
        except ValueError as exc:
            st.warning(str(exc))
    arch_start, ball_position = template_landmarks()
    use_landmarks = st.checkbox(
        f"Shape fit: use the template's landmarks as boundaries (hindfoot ends at {arch_start:.0%} of length "
        f"= start of the arch, midfoot ends at {ball_position:.0%} = ball of the foot); untick to use the % boxes above",
        value=True,
    )
    if st.button("Fit foot template (shape-based, recommended)", type="primary"):
        try:
            with st.spinner("Fitting foot outline(s)..."):
                base_cut, core_cut = strict_threshold(values)
                open_r, _ = default_morphology_radii(values.shape)
                # Thresholds only give a rough starting blob; the outline comes from the shape fit.
                seed_masks, _ = segment_feet(
                    values, threshold=base_cut, core_threshold=core_cut, n_feet=int(auto_feet), opening_radius=open_r
                )
                if not seed_masks:
                    raise ValueError("Could not locate a foot to start the fit. Crop the image or draw outlines manually.")
                fits = fit_feet(values, auto_heel, seed_masks)
            boundaries = None if use_landmarks else (hind_end / 100.0, mid_end / 100.0)
            if boundaries is not None and not boundaries[0] < boundaries[1]:
                raise ValueError("Hindfoot boundary must be smaller than the midfoot boundary.")
            region_masks, labels = [], []
            for foot_number, fit in enumerate(fits, start=1):
                parts = template_regions(fit["params"], auto_heel, values.shape, boundaries)
                for name in FOOT_REGION_NAMES:
                    region_masks.append(parts[name])
                    labels.append(f"Foot {foot_number} {name.lower()}")
            drawing_width, drawing_height = canvas_size(width, height)
            st.session_state.auto_foot = {
                "file_hash": file_hash,
                "drawing": masks_to_fabric_drawing(
                    region_masks,
                    canvas_width=drawing_width,
                    canvas_height=drawing_height,
                    original_width=width,
                    original_height=height,
                ),
                "labels": labels,
                "method": "shape template fit (scores " + ", ".join(f"{f['score']:.2f}" for f in fits) + ")",
                "feet_found": len(fits),
                "feet_requested": int(auto_feet),
            }
            st.session_state.canvas_revision += 1
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if st.button("Generate foot regions (threshold-based, less reliable)"):
        try:
            if not hind_end < mid_end:
                raise ValueError("Hindfoot boundary must be smaller than the midfoot boundary.")
            foot_masks, used_threshold = auto_segment()
            if not foot_masks:
                raise ValueError("No foot-sized warm region found. Adjust the threshold or crop the image.")
            region_masks, labels = [], []
            for foot_number, foot_mask in enumerate(foot_masks, start=1):
                parts = split_foot_regions(foot_mask, auto_heel, (hind_end / 100.0, mid_end / 100.0))
                for name in FOOT_REGION_NAMES:
                    region_masks.append(parts[name])
                    labels.append(f"Foot {foot_number} {name.lower()}")
            drawing_width, drawing_height = canvas_size(width, height)
            st.session_state.auto_foot = {
                "file_hash": file_hash,
                "drawing": masks_to_fabric_drawing(
                    region_masks,
                    canvas_width=drawing_width,
                    canvas_height=drawing_height,
                    original_width=width,
                    original_height=height,
                ),
                "labels": labels,
                "threshold": used_threshold,
                "feet_found": len(foot_masks),
                "feet_requested": int(auto_feet),
            }
            st.session_state.canvas_revision += 1
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

auto_foot = st.session_state.get("auto_foot")
if auto_foot is not None and auto_foot["file_hash"] != file_hash:
    auto_foot = None
if auto_foot is not None:
    method = auto_foot.get("method") or (
        f"threshold {auto_foot['threshold']:.4g}"
        if np.isfinite(auto_foot.get("threshold", float("nan")))
        else "split from your drawn outlines"
    )
    st.info(
        f"Foot regions loaded ({method}; {auto_foot['feet_found']} of {auto_foot['feet_requested']} feet). "
        "ROI order: " + "; ".join(f"ROI {i} = {lab}" for i, lab in enumerate(auto_foot["labels"], start=1))
        + ". Labels apply only until you add, delete or reorder contours. Check the outline and correct it with Transform if needed."
    )
    if auto_foot["feet_found"] < auto_foot["feet_requested"]:
        st.warning("Fewer feet than requested were segmented (feet may touch or be too cool); check the outlines.")

drawing_modes = {
    "Polygon": "polygon",
    "Freehand": "freedraw",
    "Rectangle": "rect",
    "Circle": "circle",
    "Transform": "transform",
}
canvas_width, canvas_height = canvas_size(width, height)
canvas_background = Image.fromarray(ensure_rgb_uint8(source_display)).resize(
    (canvas_width, canvas_height),
    Image.Resampling.NEAREST if canvas_width > width else Image.Resampling.BILINEAR,
)
canvas_result = st_canvas(
    fill_color="rgba(255, 30, 30, 0.22)",
    stroke_width=3,
    stroke_color="#00FFFF",
    background_image=canvas_background,
    update_streamlit=True,
    height=canvas_height,
    width=canvas_width,
    drawing_mode=drawing_modes[drawing_mode_label],
    initial_drawing=copy.deepcopy(auto_foot["drawing"]) if auto_foot is not None else None,
    display_toolbar=True,
    key=f"roi_canvas_{file_hash}_{st.session_state.canvas_revision}",
)

masks = masks_from_canvas(
    canvas_result.json_data,
    canvas_width=canvas_width,
    canvas_height=canvas_height,
    original_width=width,
    original_height=height,
)

if masks:
    split_help = (
        "Use on whole-foot outlines (one per foot). Each contour is divided along its own long axis "
        "from the heel into hindfoot / midfoot / forefoot using the heel side and boundaries chosen in "
        "'Automatic foot regions'. The three regions replace the outline and stay editable."
    )
    if st.button("Split each drawn contour into hindfoot / midfoot / forefoot", help=split_help):
        try:
            if not hind_end < mid_end:
                raise ValueError("Hindfoot boundary must be smaller than the midfoot boundary.")
            region_masks, labels = [], []
            for contour_number, contour_mask in enumerate(masks, start=1):
                parts = split_foot_regions(contour_mask, auto_heel, (hind_end / 100.0, mid_end / 100.0))
                for name in FOOT_REGION_NAMES:
                    region_masks.append(parts[name])
                    labels.append(f"Contour {contour_number} {name.lower()}")
            st.session_state.auto_foot = {
                "file_hash": file_hash,
                "drawing": masks_to_fabric_drawing(
                    region_masks,
                    canvas_width=canvas_width,
                    canvas_height=canvas_height,
                    original_width=width,
                    original_height=height,
                ),
                "labels": labels,
                "threshold": float("nan"),
                "feet_found": len(masks),
                "feet_requested": len(masks),
            }
            st.session_state.canvas_revision += 1
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

rows = []
whole_stats = summary_statistics(values)
rows.append(
    {
        "region": "Whole image",
        **whole_stats,
        "roi_pixels": int(values.size),
        "roi_fraction": 1.0,
        "area": width * height * row_spacing * column_spacing,
        "perimeter": 2 * (height * row_spacing + width * column_spacing),
        "equivalent_diameter": np.sqrt(
            4 * width * height * row_spacing * column_spacing / np.pi
        ),
        "bbox_x_min": 0,
        "bbox_y_min": 0,
        "bbox_x_max": width - 1,
        "bbox_y_max": height - 1,
    }
)
for index, mask in enumerate(masks, start=1):
    rows.append(
        {
            "region": f"ROI {index}",
            **summary_statistics(values, mask),
            **roi_geometry(mask, row_spacing=row_spacing, column_spacing=column_spacing),
        }
    )

combined_mask = None
if masks:
    combined_mask = np.logical_or.reduce(masks)
    if len(masks) > 1:
        rows.append(
            {
                "region": "Combined ROI",
                **summary_statistics(values, combined_mask),
                **roi_geometry(
                    combined_mask,
                    row_spacing=row_spacing,
                    column_spacing=column_spacing,
                ),
            }
        )

results = pd.DataFrame(rows)
results.insert(1, "measurement", measurement_name)
results.insert(2, "measurement_unit", unit)
results["area_unit"] = f"{spatial_unit}²"
results["perimeter_unit"] = spatial_unit

st.subheader("Results")
metric_columns = st.columns(3)
metric_columns[0].metric(
    f"Whole-image mean{f' ({unit})' if unit else ''}",
    f"{whole_stats['mean']:.4g}" if np.isfinite(whole_stats["mean"]) else "NA",
)
metric_columns[1].metric("Contours", len(masks))
if combined_mask is not None:
    combined_stats = summary_statistics(values, combined_mask)
    metric_columns[2].metric(
        f"Contour mean{f' ({unit})' if unit else ''}",
        f"{combined_stats['mean']:.4g}" if np.isfinite(combined_stats["mean"]) else "NA",
    )
else:
    metric_columns[2].metric("Contour mean", "Draw a contour")

# On-screen table is deliberately compact; the exported statistics CSV/ZIP keep every column.
displayed_columns = ["region", "mean", "median", "p95", "min", "max"]
st.dataframe(results[displayed_columns], use_container_width=True, hide_index=True)

comparison_rows: list[dict] = []
comparison_threshold = None
if len(masks) >= 2:
    st.subheader("Left-right comparison")
    st.caption(
        "Pairs two ROIs and reports A minus B for mean, median and p95, plus the areas. For foot regions "
        "created by the app the pairs are filled in automatically (the same region on the first two feet); "
        "after you add, delete or reorder contours, check the ROI numbers. Foot 1 is the foot on the left of "
        "the image: decide yourself which side that is."
    )
    auto_labels = (st.session_state.get("auto_foot") or {}).get("labels")
    if (st.session_state.get("auto_foot") or {}).get("file_hash") != file_hash:
        auto_labels = None
    pair_defaults = pd.DataFrame(
        [{"Region": label, "ROI A": a, "ROI B": b} for label, a, b in default_region_pairs(auto_labels, len(masks))]
    )
    pair_table = st.data_editor(
        pair_defaults,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        key=f"lr_pairs_{file_hash}_{st.session_state.canvas_revision}_{len(masks)}",
        column_config={
            "Region": st.column_config.TextColumn("Region / label"),
            "ROI A": st.column_config.NumberColumn("ROI A", min_value=1, max_value=len(masks), step=1),
            "ROI B": st.column_config.NumberColumn("ROI B", min_value=1, max_value=len(masks), step=1),
        },
    )
    threshold_text = st.number_input(
        "Optional: flag pairs where |mean difference| is at least (0 = off)",
        min_value=0.0,
        value=0.0,
        step=0.1,
        help="A screening aid only. 2.2 C (4 F) is the value commonly used for contralateral plantar "
        "differences in diabetic feet, but studies question it as a single-measurement warning (see README).",
    )
    comparison_threshold = float(threshold_text) if threshold_text > 0 else None
    valid_pairs = []
    for _, pair in pair_table.iterrows():
        if pd.isna(pair["ROI A"]) or pd.isna(pair["ROI B"]):
            continue
        valid_pairs.append((str(pair["Region"] or f"ROI {int(pair['ROI A'])} vs {int(pair['ROI B'])}"),
                            int(pair["ROI A"]), int(pair["ROI B"])))
    try:
        comparison_rows = left_right_comparison(
            values, masks, valid_pairs, row_spacing=row_spacing, column_spacing=column_spacing,
            threshold=comparison_threshold,
        )
    except ValueError as exc:
        st.warning(str(exc))
    if comparison_rows:
        comparison = pd.DataFrame(comparison_rows)
        shown = comparison[
            ["region", "roi_a", "roi_b", "mean_a", "mean_b", "mean_difference", "median_difference",
             "p95_difference", "area_a", "area_b", "area_difference_percent"]
            + (["exceeds_threshold"] if comparison_threshold is not None else [])
        ].rename(columns={"roi_a": "ROI A", "roi_b": "ROI B"})
        st.dataframe(shown, use_container_width=True, hide_index=True)
        if (comparison["valid_pixels_a"] == 0).any() or (comparison["valid_pixels_b"] == 0).any():
            st.warning("At least one ROI has no valid (decoded) pixels, so its differences are NA.")
        st.caption(f"Area unit: {spatial_unit}². Difference = A minus B (area difference is the symmetric percentage).")

if masks:
    st.subheader("Contour quality check")
    overlay = overlay_masks(source_display, masks)
    show_image(overlay, caption="Verify that every colored outline matches the intended region.")
else:
    overlay = ensure_rgb_uint8(source_display)
    st.info("Draw at least one closed contour to calculate contour-specific measurements.")

region_options = list(results["region"])
selected_region = st.selectbox("Distribution to inspect", region_options)
if selected_region == "Whole image":
    distribution = values[np.isfinite(values)]
elif selected_region == "Combined ROI" and combined_mask is not None:
    distribution = values[combined_mask & np.isfinite(values)]
else:
    roi_number = int(selected_region.split()[-1]) - 1
    distribution = values[masks[roi_number] & np.isfinite(values)]
if distribution.size:
    histogram, edges = np.histogram(distribution, bins=min(50, max(10, int(np.sqrt(distribution.size)))))
    chart = pd.DataFrame(
        {
            "value": (edges[:-1] + edges[1:]) / 2,
            "count": histogram,
        }
    ).set_index("value")
    st.bar_chart(chart)

st.subheader("Export")
include_value_map = st.checkbox(
    "Include the full calibrated measurement map as CSV (can make the download large)",
    value=False,
)
base = sanitized_stem(uploaded.name)
statistics_csv = results.to_csv(index=False).encode("utf-8")
contour_json = {
    "source_filename": uploaded.name,
    "source_sha256_prefix": file_hash,
    "source_width": width,
    "source_height": height,
    "canvas_width": canvas_width,
    "canvas_height": canvas_height,
    "fabric_canvas": canvas_result.json_data or {"objects": []},
}
metadata = {
    "source_filename": uploaded.name,
    "source_type": source_kind,
    "source_metadata": source_metadata,
    "measurement_name": measurement_name,
    "measurement_unit": unit,
    "value_remapping": {
        "applied": bool(apply_remap),
        "source_minimum": data_min,
        "source_maximum": data_max,
        "mapped_minimum": float(target_min) if apply_remap else None,
        "mapped_maximum": float(target_max) if apply_remap else None,
    },
    "spatial_calibration": {
        "row_spacing": row_spacing,
        "column_spacing": column_spacing,
        "distance_unit": spatial_unit,
    },
    "false_color_decoding_is_approximate": approximate_decode,
    "false_color_decoding": decode_details,
    "contour_count": len(masks),
    "left_right_comparison": {
        "pairs": [[row["region"], row["roi_a"], row["roi_b"]] for row in comparison_rows],
        "threshold": comparison_threshold,
        "convention": "difference = A minus B",
    },
}

archive_buffer = io.BytesIO()
with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.writestr(f"{base}_statistics.csv", statistics_csv)
    archive.writestr(f"{base}_contours.json", json.dumps(contour_json, indent=2))
    archive.writestr(f"{base}_metadata.json", json.dumps(metadata, indent=2))
    if comparison_rows:
        archive.writestr(
            f"{base}_left_right_comparison.csv",
            pd.DataFrame(comparison_rows).to_csv(index=False).encode("utf-8"),
        )
    archive.writestr(f"{base}_overlay.png", image_bytes(overlay))
    if combined_mask is not None:
        archive.writestr(f"{base}_combined_mask.png", mask_bytes(combined_mask))
        for index, mask in enumerate(masks, start=1):
            archive.writestr(f"{base}_roi_{index}_mask.png", mask_bytes(mask))
    if include_value_map:
        value_map_buffer = io.StringIO()
        pd.DataFrame(values).to_csv(value_map_buffer, index=False, header=False)
        archive.writestr(f"{base}_calibrated_value_map.csv", value_map_buffer.getvalue())

download_columns = st.columns(3)
download_columns[0].download_button(
    "Download complete analysis (.zip)",
    data=archive_buffer.getvalue(),
    file_name=f"{base}_roi_analysis.zip",
    mime="application/zip",
    use_container_width=True,
)
download_columns[1].download_button(
    "Download statistics (.csv)",
    data=statistics_csv,
    file_name=f"{base}_statistics.csv",
    mime="text/csv",
    use_container_width=True,
)
download_columns[2].download_button(
    "Download overlay (.png)",
    data=image_bytes(overlay),
    file_name=f"{base}_overlay.png",
    mime="image/png",
    use_container_width=True,
)

st.divider()
st.caption(
    "Quantitative note: ordinary PNG/JPEG screenshots contain colors or intensities, not necessarily "
    "the original physiological values. For clinical or research measurements, prefer raw device "
    "exports, calibrated CSV matrices, or DICOM data with valid scaling metadata."
)
