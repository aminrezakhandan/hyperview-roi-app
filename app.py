"""Streamlit app for interactive analysis of imaging and pixel-map files."""

from __future__ import annotations

import base64
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

from roi_core import (
    SUPPORTED_COLORMAPS,
    decode_false_color,
    ensure_rgb_uint8,
    masks_from_canvas,
    normalize_to_rgb,
    overlay_masks,
    roi_geometry,
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


def read_raster(payload: bytes) -> tuple[np.ndarray, np.ndarray, dict]:
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(payload)))
    metadata = {
        "format": image.format,
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
    }
    if image.mode in {"L", "I", "I;16", "F"}:
        scalar = np.asarray(image, dtype=float)
        display = normalize_to_rgb(scalar)
        return scalar, display, metadata
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
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
        values = np.full_like(values, float(target_min), dtype=float)
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

height, width = values.shape
st.write(
    f"**Source:** {uploaded.name} · **Type:** {source_kind} · "
    f"**Measurement grid:** {width} × {height} pixels"
)

with st.expander("Source details"):
    st.json(source_metadata)

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
        st.rerun()

drawing_modes = {
    "Polygon": "polygon",
    "Freehand": "freedraw",
    "Rectangle": "rect",
    "Circle": "circle",
    "Transform": "transform",
}
canvas_width = min(900, width)
canvas_height = max(1, int(round(height * canvas_width / width)))
canvas_background = Image.fromarray(ensure_rgb_uint8(source_display)).resize(
    (canvas_width, canvas_height),
    Image.Resampling.BILINEAR,
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

rows = []
whole_stats = summary_statistics(values)
rows.append(
    {
        "region": "Whole image",
        **whole_stats,
        "roi_pixels": int(np.isfinite(values).size),
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

preferred_columns = [
    "region",
    "mean",
    "median",
    "std",
    "min",
    "p05",
    "p25",
    "p75",
    "p95",
    "max",
    "valid_pixels",
    "missing_pixels",
    "valid_fraction",
    "roi_pixels",
    "roi_fraction",
    "area",
    "perimeter",
    "equivalent_diameter",
]
st.dataframe(results[preferred_columns], use_container_width=True, hide_index=True)

if masks:
    st.subheader("Contour quality check")
    overlay = overlay_masks(source_display, masks)
    st.image(
        overlay,
        caption="Verify that every colored outline matches the intended region.",
        use_container_width=True,
    )
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
}

archive_buffer = io.BytesIO()
with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.writestr(f"{base}_statistics.csv", statistics_csv)
    archive.writestr(f"{base}_contours.json", json.dumps(contour_json, indent=2))
    archive.writestr(f"{base}_metadata.json", json.dumps(metadata, indent=2))
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
