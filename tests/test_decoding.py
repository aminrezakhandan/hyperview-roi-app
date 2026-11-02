"""Synthetic-image tests for false-color decoding and ROI statistics.

Only generated colormap gradients and arrays are used; no real or patient images.
"""

import io

import numpy as np
import pytest
from matplotlib import colormaps
from PIL import Image

from roi_core import (
    SUPPORTED_COLORMAPS,
    decode_false_color,
    ensure_rgb_uint8,
    flatten_alpha,
    roi_geometry,
    summary_statistics,
)

VMIN, VMAX = 20.0, 40.0  # e.g. a thermal scale in degrees C
WIDTH = 1000


def gradient(cmap, reverse=False, width=WIDTH, rows=3):
    positions = np.linspace(0, 1, width)
    if reverse:
        positions = positions[::-1]
    rgb = colormaps[cmap](positions, bytes=True)[..., :3]
    truth = np.tile(np.linspace(VMIN, VMAX, width), (rows, 1))
    return np.tile(rgb[None], (rows, 1, 1)), truth


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("cmap", SUPPORTED_COLORMAPS)
def test_lossless_gradient_recovered(cmap, reverse):
    rgb, truth = gradient(cmap, reverse)
    values, valid, dist = decode_false_color(rgb, cmap, VMIN, VMAX, reverse=reverse)
    assert valid.all()
    # LUT step is range/255 = 0.078; allow ~2 steps for local colormap ambiguity.
    assert np.abs(values - truth).max() < 0.25
    assert np.abs(values - truth).mean() < 0.05
    assert dist.max() < 10


@pytest.mark.parametrize("cmap", ["turbo", "jet", "viridis"])
def test_wrong_direction_is_detected_by_value_error(cmap):
    rgb, truth = gradient(cmap)
    values, _, _ = decode_false_color(rgb, cmap, VMIN, VMAX, reverse=True)
    assert np.abs(values - truth).max() > 15  # reverse flag really flips the scale


def test_swapped_min_max_inverts_scale():
    rgb, truth = gradient("turbo")
    values, _, _ = decode_false_color(rgb, "turbo", VMAX, VMIN)
    assert np.abs(values - truth[:, ::-1]).max() < 0.25


def test_negative_and_wide_ranges():
    rgb, _ = gradient("jet")
    values, _, _ = decode_false_color(rgb, "jet", -50.0, 350.0)
    assert values.min() == pytest.approx(-50.0, abs=0.01)
    assert values.max() == pytest.approx(350.0, abs=0.01)


def test_jpeg_compression_error_is_bounded_but_nonzero():
    """Documents the screenshot caveat: lossy compression adds error."""
    rgb, truth = gradient("turbo", rows=32)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=75)
    jpeg = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"))
    values, valid, _ = decode_false_color(jpeg, "turbo", VMIN, VMAX, max_rgb_distance=None)
    assert valid.all()
    err = np.abs(values - truth)
    assert err.mean() < 0.5  # well inside 1/40 of the range on a 20-degree scale
    assert err.max() > 0.075  # but not lossless


def test_out_of_range_colors_masked_and_nan():
    rgb, _ = gradient("jet", width=50)
    rgb = rgb.copy()
    rgb[0, 0] = (255, 0, 255)  # magenta is not on the jet scale
    values, valid, dist = decode_false_color(rgb, "jet", 0, 1, max_rgb_distance=45)
    assert not valid[0, 0] and np.isnan(values[0, 0])
    assert dist[0, 0] > 45
    assert valid.sum() == valid.size - 1


def test_threshold_none_keeps_everything():
    rgb = np.full((2, 2, 3), (255, 0, 255), dtype=np.uint8)
    values, valid, _ = decode_false_color(rgb, "jet", 0, 1, max_rgb_distance=None)
    assert valid.all() and np.isfinite(values).all()


def test_black_background_pitfall_documented():
    """Black pixels decode as the minimum for colormaps that start near black.

    This is a known limitation (see README); the mismatch threshold does not help
    for inferno/magma/hot, so legends/backgrounds must be excluded by the ROI.
    """
    black = np.zeros((2, 2, 3), dtype=np.uint8)
    for cmap in ("inferno", "magma", "hot"):
        values, valid, _ = decode_false_color(black, cmap, 0, 100)
        assert valid.all() and values.max() == pytest.approx(0.0)
    for cmap in ("jet", "turbo", "viridis", "rainbow"):
        _, valid, _ = decode_false_color(black, cmap, 0, 100)
        assert not valid.any()


def test_rgba_and_grayscale_inputs_accepted():
    rgb, truth = gradient("turbo")
    rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], axis=2)
    values, valid, _ = decode_false_color(rgba, "turbo", VMIN, VMAX)
    assert valid.all() and np.abs(values - truth).max() < 0.25
    gray = np.tile(np.linspace(0, 255, 16), (4, 1))
    assert decode_false_color(gray, "turbo", 0, 1, max_rgb_distance=None)[0].shape == (4, 16)


def test_float_rgb_in_unit_range_is_not_truncated():
    rgb, truth = gradient("turbo")
    values, valid, _ = decode_false_color(rgb / 255.0, "turbo", VMIN, VMAX)
    assert valid.all() and np.abs(values - truth).max() < 0.25
    assert ensure_rgb_uint8(np.full((2, 2, 3), 0.5)).max() in (127, 128)


def test_dark_uint8_image_not_rescaled():
    dark = np.ones((2, 2, 3), dtype=np.uint8)
    assert (ensure_rgb_uint8(dark) == 1).all()


def test_transparent_pixels_flattened_to_white_not_black():
    img = Image.new("RGBA", (2, 1), (0, 0, 0, 0))  # fully transparent black
    out = flatten_alpha(img)
    assert (out == 255).all()
    _, valid, _ = decode_false_color(out, "inferno", 0, 100)
    assert not valid.any()  # white is rejected, whereas transparent-black decoded as 0
    opaque = Image.new("RGBA", (1, 1), (10, 20, 30, 255))
    assert tuple(flatten_alpha(opaque)[0, 0]) == (10, 20, 30)


def test_palette_image_with_transparency():
    img = Image.new("P", (2, 1))
    img.putpalette([0, 0, 0, 255, 0, 0] + [0] * 756)
    img.info["transparency"] = 0
    img.putpixel((1, 0), 1)
    out = flatten_alpha(img)
    assert tuple(out[0, 0]) == (255, 255, 255) and tuple(out[0, 1]) == (255, 0, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cmap_name": "nope"},
        {"value_min": float("nan")},
        {"value_max": float("inf")},
        {"lut_size": 1},
    ],
)
def test_invalid_arguments_raise(kwargs):
    rgb, _ = gradient("jet", width=8)
    args = {"cmap_name": "jet", "value_min": 0.0, "value_max": 1.0, **kwargs}
    with pytest.raises(ValueError):
        decode_false_color(rgb, args.pop("cmap_name"), args.pop("value_min"), args.pop("value_max"), **args)


def test_empty_image_raises():
    with pytest.raises(ValueError):
        decode_false_color(np.zeros((0, 4, 3), dtype=np.uint8), "jet", 0, 1)


def test_unsupported_shape_raises():
    with pytest.raises(ValueError):
        ensure_rgb_uint8(np.zeros((4, 4, 2)))


def test_large_image_decodes_quickly_and_matches_small():
    rgb, truth = gradient("turbo", width=2000, rows=1500)  # 3 MP, 2000 unique colours
    values, valid, _ = decode_false_color(rgb, "turbo", VMIN, VMAX)
    assert valid.all() and np.abs(values - truth).max() < 0.25


def test_single_pixel_image():
    values, valid, _ = decode_false_color(np.array([[[0, 0, 128]]], dtype=np.uint8), "jet", 0, 10)
    assert valid[0, 0] and values[0, 0] == pytest.approx(0.0, abs=0.05)


# ---- ROI statistics -------------------------------------------------------


def test_stats_known_values():
    data = np.arange(1.0, 11.0).reshape(2, 5)
    s = summary_statistics(data)
    assert s["mean"] == pytest.approx(5.5)
    assert s["median"] == pytest.approx(5.5)
    assert s["std"] == pytest.approx(np.std(np.arange(1, 11), ddof=1))
    assert (s["min"], s["max"]) == (1.0, 10.0)
    assert s["valid_fraction"] == 1.0


def test_stats_empty_roi_is_nan_not_error():
    s = summary_statistics(np.ones((3, 3)), np.zeros((3, 3), dtype=bool))
    assert s["selected_pixels"] == 0 and s["valid_pixels"] == 0
    assert np.isnan(s["mean"]) and np.isnan(s["valid_fraction"])


def test_stats_all_nan_roi():
    s = summary_statistics(np.full((2, 2), np.nan))
    assert s["valid_pixels"] == 0 and s["missing_pixels"] == 4 and np.isnan(s["median"])


def test_stats_single_pixel_std_zero():
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    s = summary_statistics(np.arange(9.0).reshape(3, 3), mask)
    assert s["mean"] == 4.0 and s["std"] == 0.0


def test_stats_excludes_decoded_invalid_pixels_from_denominator_of_mean():
    rgb, _ = gradient("jet", width=10, rows=1)
    rgb = rgb.copy()
    rgb[0, 0] = (255, 0, 255)
    values, _, _ = decode_false_color(rgb, "jet", 0, 9)
    s = summary_statistics(values)
    assert s["selected_pixels"] == 10 and s["valid_pixels"] == 9 and s["missing_pixels"] == 1


def test_stats_shape_validation():
    with pytest.raises(ValueError):
        summary_statistics(np.ones((2, 2)), np.ones((3, 3), dtype=bool))
    with pytest.raises(ValueError):
        summary_statistics(np.ones((2, 2, 3)))


def test_geometry_empty_and_l_shape():
    empty = roi_geometry(np.zeros((4, 4), dtype=bool))
    assert empty["roi_pixels"] == 0 and empty["area"] == 0.0
    mask = np.zeros((4, 4), dtype=bool)
    mask[0:2, 0:2] = True
    mask[2, 0] = True  # L-shape: 5 px, perimeter 10
    g = roi_geometry(mask)
    assert g["roi_pixels"] == 5 and g["perimeter"] == 10.0
    assert (g["bbox_x_min"], g["bbox_y_min"], g["bbox_x_max"], g["bbox_y_max"]) == (0, 0, 1, 2)
