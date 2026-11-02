# Multimodal Imaging ROI Analyzer

A Streamlit app (runs locally or on Streamlit Community Cloud) that accepts:

- PNG, JPEG, TIFF, and BMP images
- DICOM images, including common multiframe datasets
- CSV pixel maps stored either as a numeric matrix or as `x, y, value` rows

It calculates whole-image statistics, lets the user draw multiple polygon,
freehand, rectangular, or circular contours, and calculates the same statistics
plus area, perimeter, equivalent diameter, and bounding box for each contour and
their combined region.

## Installation

Use Python 3.10 or newer. From this folder, run:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Then install and start the app:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The browser should open automatically. If it does not, open the local address
shown in the terminal, usually `http://localhost:8501`.

## Recommended workflow

1. Upload one image, DICOM file, or CSV pixel map.
2. Name the measurement and enter its unit, such as `StO2 (%)` or
   `Temperature (°C)`.
3. If needed, open **Optional value remapping** and enter the image/colorbar
   min and max (for example temperature limits), then apply remapping.
4. Enter spatial calibration or retain DICOM pixel spacing when available.
5. Select **Polygon**, click around the region, and double-click the final point.
6. Verify the contour in the quality-check overlay.
7. Download the ZIP package containing statistics, contour definitions, masks,
   the overlay, and optional calibrated values.

Freehand contours are automatically closed. Polygon contours are preferred for
repeatability and easier visual verification.

## Supported colormaps (false-color decoding)

`turbo`, `jet`, `rainbow`, `inferno`, `magma`, `plasma`, `viridis`, `hot`,
`coolwarm` (Matplotlib definitions), with an optional **Reverse colormap**
switch. Decoding maps each pixel to the nearest colour on a 256-step lookup
table of the chosen colormap and linearly rescales that position to the
displayed minimum/maximum you enter. Pixels farther than the **Maximum RGB
mismatch** threshold (default 45, on a 0-441 scale) from any colormap colour are
set to missing (NaN) and excluded from statistics.

## Accuracy caveat: screenshots vs raw exports

Decoding a false-color picture is an **estimate**, not a measurement.

- On lossless synthetic gradients of the exact same colormap the decoder
  recovers values to within about one LUT step (range/255; tests assert < 0.25
  on a 20-degree scale). Real screenshots do much worse: a vendor palette that
  is not exactly the Matplotlib one, JPEG compression, scaling/anti-aliasing,
  and rendered overlays/text shift colours. Expect errors on the order of
  **~2 C** (or ~2-3 % of the displayed range) for thermal screenshots, and that
  error is **not reduced** by averaging over an ROI because it is largely
  systematic (colormap mismatch).
- The displayed minimum/maximum must be typed in correctly; a wrong range
  scales every value.
- Colours near black or white (backgrounds, legends, text, ROI outlines) can
  decode as real values - black decodes as the **minimum** for `inferno`,
  `magma` and `hot`, and white as the maximum for `hot`. Keep contours inside
  the thermal/StO2 area only.
- Transparent PNG regions are composited on white before decoding.
- Prefer raw radiometric exports (FLIR CSV/radiometric JPEG/TIFF, HyperView
  data export, DICOM with rescale metadata) whenever available; use the
  screenshot decoder only for exploratory or sensitivity analyses and report it
  as such.

## What is quantitatively valid?

- **CSV numeric matrix:** Quantitative when the exported values and units are
  documented by the device/software.
- **DICOM:** Quantitative when pixel data, rescale metadata, and units are valid.
- **Ordinary PNG/JPEG:** The app can always report pixel intensity or RGB-channel
  statistics. Those numbers are not automatically temperature, oxygenation, or
  hemoglobin measurements.
- **False-color screenshot:** The optional decoder estimates values only when
  the selected colormap and displayed minimum/maximum exactly match the source
  image. Compression, labels, legends, overlays, and an incorrect colormap
  introduce error. Prefer raw HyperView, Kent, or thermal-camera exports for
  research analysis.

## Privacy and data handling

Uploaded files are processed in memory in the Streamlit process; the app does
not write uploads to disk, log them, or keep them after the session. Exports are
generated on demand and downloaded by the user. **Do not upload patient data or
anything containing protected health information** to a public deployment
(filenames and DICOM headers can contain identifiers). For identifiable data run
the app locally under your institution's approval.

Limits: raster images over 40 megapixels are rejected to bound memory use.

## Test

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
```

The tests use only synthetic arrays and generated colormap gradients.
