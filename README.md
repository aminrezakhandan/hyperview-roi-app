# Multimodal Imaging ROI Analyzer

A Streamlit app (runs locally or on Streamlit Community Cloud) that accepts:

- PNG, JPEG, TIFF, and BMP images
- DICOM images, including common multiframe datasets
- CSV pixel maps stored either as a numeric matrix or as `x, y, value` rows
  (text header rows/label columns such as `File:` / `Frame 1` in camera exports
  are trimmed automatically)

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

## Viewing two images side by side

Use **Compare** in the sidebar to add a second image or CSV matrix. It is shown
next to the main file under *Side-by-side comparison*. This is view only (no
contours or statistics are computed for the second file), and CSV maps are
colour-scaled separately, so identical colours do not mean identical values.

## Results table and exports

The on-screen results table shows `region, mean, median, p95, min, max`. The
downloaded statistics CSV and ZIP keep every detail (all percentiles, std, pixel
counts, area, perimeter, bounding box, metadata, masks, overlay).

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

Contours are rasterized directly on the source-resolution pixel grid: a pixel
is inside an ROI when its centre lies inside the drawn shape. ROI area is
`pixels x row spacing x column spacing`, accurate to about +/-1 pixel along the
boundary (curved or diagonal edges are unbiased; shapes whose edges fall exactly
on pixel centres can differ by about half a boundary pixel). The **Combined ROI**
is the union, so overlapping contours are not double counted. Perimeter is
measured along pixel edges and therefore overestimates diagonal and curved
boundaries (about 27 % for a circle); use area and equivalent diameter for size.

Freehand contours are automatically closed. Polygon contours are preferred for
repeatability and easier visual verification.

## Automatic foot regions (plantar thermal, exploratory)

Open **Automatic foot regions** above the canvas, choose how many feet are in
the image and which side of the image the heel is on, then press **Generate foot
regions**. The app:

1. keeps the warmest connected area(s) (Otsu threshold, or a manual threshold),
   fills holes and ignores small blobs (< 0.5 % of the image);
2. finds each foot's long axis (principal axis of its pixels);
3. splits the projected length from the heel into hindfoot / midfoot / forefoot
   (default 0-33 % / 33-66 % / 66-100 %, adjustable), so the three regions are
   disjoint and exactly cover the foot mask;
4. loads them onto the canvas as ordinary polygons, so they can be dragged,
   corrected or deleted, and they flow through the same statistics and export.

**Shape-based fit (recommended): "Fit foot template".** A generic plantar foot
outline (heel, arch, ball, toes) is fitted to each foot by position, length,
rotation (+/-45 degrees), width, arch fullness and left/right mirroring. The fit
maximises how strongly temperature falls off across the outline plus
inside-versus-surrounding contrast, so it does not depend on a single
temperature cut-off and warm bedding or cool toes matter much less. (A rough
threshold blob is used only as the starting guess.) With two feet the pair is
forced to opposite mirror states. Region boundaries come from the template's
landmarks: the hindfoot ends at the start of the arch (about 29 % of length from
the heel) and the midfoot ends at the ball of the foot, its widest point (about
72 %); untick the option to use your own percentages. The regions load onto the
canvas as editable polygons. **Limits:** the template is a hand-designed generic
shape, not anthropometric data, so unusual feet (very flat or high arch,
deformity, amputation, toes curled or cut off by the image) and oblique views
will not fit exactly, and cool toe tips are sometimes missed by a few pixels.
Correct with Transform, or redraw. Validate on your own images (for example Dice
overlap against manual contours) before relying on it.

**Alternative: draw, then split.** Automatic segmentation can fail when warm
bedding is as hot as cooler parts of the foot (heels, toes). Instead, draw one
rough outline around each foot (Polygon tool), choose the heel side and
boundaries, and press **Split each drawn contour into hindfoot / midfoot /
forefoot** under the canvas. Each outline is divided along its own long axis, the
three regions exactly cover your outline, and they stay editable. The result is
only as accurate as your outline. Tick **Preview the automatic foot mask** to see
what the automatic modes would select before using them.

**Threshold modes:** *Automatic (Otsu)* keeps everything warmer than the
background/foot split; *Strict (exclude warm bedding)* seeds from a higher
two-level Otsu threshold, cuts thin bridges (morphological opening), then grows
back a few pixels over the Otsu mask. Use Strict when warm bedding or clothing
touches the feet; it can trim cool toe tips, so check the forefoot outline.

This is pure NumPy/SciPy: no model, no network calls, no tokens, and it runs in
well under a second. Limits: it assumes a roughly upright plantar view with the
foot warmer than its surroundings, and splitting by fractions of length is an
approximation of the anatomical boundaries, not a landmark-based one. Legends,
hands, bedding or warm objects touching the foot, and feet that touch each other,
can break the segmentation, so crop first and always verify the outlines. It has
only been tested on synthetic shapes; validate it on your own images (for
example Dice overlap against manual contours) before relying on it, and report
results from it as exploratory.

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
  and rendered overlays/text shift colours. The project's experience is errors on the order of
  **~2 C** for thermal screenshots (not a validated bound), and that
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
