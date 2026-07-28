# Multimodal Imaging ROI Analyzer

This local Streamlit app accepts:

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

## Privacy

The app performs analysis in the Python process where it is run. Running it
locally does not intentionally transmit uploaded files. Avoid deploying it to a
shared server without appropriate access controls and institutional approval,
especially when DICOM files may contain protected health information.

## Test

After installing the requirements:

```bash
python -m unittest test_roi_core.py
```
