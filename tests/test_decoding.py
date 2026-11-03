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


# ---- Contour -> mask area accuracy ----------------------------------------

from roi_core import masks_from_canvas, object_to_polygon  # noqa: E402


def _area(obj, cw, ch, ow, oh):
    masks = masks_from_canvas({"objects": [obj]}, canvas_width=cw, canvas_height=ch,
                              original_width=ow, original_height=oh)
    return int(masks[0].sum()) if masks else 0


def test_rect_area_exact_at_source_resolution():
    obj = {"type": "rect", "left": 10, "top": 10, "width": 40, "height": 40}
    assert _area(obj, 100, 100, 100, 100) == 1600  # was 1681 (boundary pixels counted)
    assert _area(obj, 900, 900, 1800, 1800) == 6400  # 2x upscale: 80 x 80


def test_non_square_scaling_uses_separate_axes():
    obj = {"type": "rect", "left": 0, "top": 0, "width": 50, "height": 20}
    assert _area(obj, 100, 40, 300, 200) == 150 * 100


def test_circle_area_within_half_percent():
    obj = {"type": "circle", "left": 200, "top": 200, "width": 200, "height": 200,
           "originX": "center", "originY": "center"}
    area = _area(obj, 400, 400, 400, 400)
    assert area == pytest.approx(np.pi * 100**2, rel=0.005)


def test_rotated_rect_and_scaled_ellipse_area():
    rect = {"type": "rect", "left": 100, "top": 100, "width": 100, "height": 40, "angle": 37,
            "originX": "center", "originY": "center"}
    assert _area(rect, 300, 300, 300, 300) == pytest.approx(4000, rel=0.02)
    ell = {"type": "circle", "left": 150, "top": 150, "width": 50, "height": 50, "scaleX": 2, "scaleY": 1,
           "originX": "center", "originY": "center"}
    assert _area(ell, 300, 300, 300, 300) == pytest.approx(np.pi * 50 * 25, rel=0.01)


def test_polygon_area_matches_shoelace_for_random_shapes():
    rng = np.random.default_rng(0)
    for _ in range(20):
        # random star-shaped polygon, irrational-ish coordinates (no pixel-centre ties)
        n = int(rng.integers(5, 12))
        # evenly spaced angles with jitter < half a step keeps the polygon simple (star-shaped)
        ang = (np.arange(n) + rng.uniform(-0.4, 0.4, n)) * 2 * np.pi / n
        rad = rng.uniform(30, 90, n)
        pts = np.column_stack([150 + rad * np.cos(ang), 150 + rad * np.sin(ang)])
        x, y = pts[:, 0], pts[:, 1]
        shoelace = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        mx, my = x.min(), y.min()
        obj = {"type": "polygon", "left": mx, "top": my, "width": x.max() - mx, "height": y.max() - my,
               "points": [{"x": float(a - mx), "y": float(b - my)} for a, b in pts]}
        assert _area(obj, 300, 300, 300, 300) == pytest.approx(shoelace, rel=0.01)


def test_freehand_path_curves_are_flattened():
    # Quadratic strokes: control points at the corners of a square, end points at midpoints.
    obj = {"type": "path", "left": 0, "top": 0, "width": 100, "height": 100,
           "path": [["M", 50, 0], ["Q", 100, 0, 100, 50], ["Q", 100, 100, 50, 100],
                    ["Q", 0, 100, 0, 50], ["Q", 0, 0, 50, 0]]}
    # Rounded diamond: area = 5000 + curved bulge, strictly between diamond and square.
    area = _area(obj, 100, 100, 100, 100)
    assert 5000 < area < 10000 and area == pytest.approx(7000 - 0, abs=1500)
    assert len(object_to_polygon(obj)) > 20


def test_overlapping_contours_not_double_counted_in_union():
    a = {"type": "rect", "left": 0, "top": 0, "width": 40, "height": 40}
    b = {"type": "rect", "left": 20, "top": 20, "width": 40, "height": 40}
    masks = masks_from_canvas({"objects": [a, b]}, canvas_width=100, canvas_height=100,
                              original_width=100, original_height=100)
    union = np.logical_or.reduce(masks)
    assert [int(m.sum()) for m in masks] == [1600, 1600] and int(union.sum()) == 1600 + 1600 - 400


def test_contour_partly_outside_image_is_clipped_and_degenerate_ignored():
    out = {"type": "rect", "left": 90, "top": 90, "width": 40, "height": 40}
    assert _area(out, 100, 100, 100, 100) == 100
    assert _area({"type": "rect", "left": 5, "top": 5, "width": 0, "height": 10}, 100, 100, 100, 100) == 0
    assert _area({"type": "rect", "left": 200, "top": 200, "width": 10, "height": 10}, 100, 100, 100, 100) == 0
    assert _area({"type": "line"}, 100, 100, 100, 100) == 0


# ---- Automatic foot regions (synthetic shapes only) -----------------------

from roi_core import (  # noqa: E402
    FOOT_REGION_NAMES,
    masks_to_fabric_drawing,
    otsu_threshold,
    segment_feet,
    split_foot_regions,
)


def synthetic_foot(h=300, w=200, cx=100, cy=150, ry=120, rx=45, angle=0.0, warm=33.0, cold=22.0, noise=0.0, seed=0):
    """Elongated warm ellipse (toes up, heel at bottom) on a cool background."""
    yy, xx = np.mgrid[0:h, 0:w]
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    u = (xx - cx) * c + (yy - cy) * s
    v = -(xx - cx) * s + (yy - cy) * c
    inside = (u / rx) ** 2 + (v / ry) ** 2 <= 1
    img = np.where(inside, warm, cold).astype(float)
    if noise:
        img = img + np.random.default_rng(seed).normal(0, noise, img.shape)
    return img, inside


def test_otsu_separates_two_levels():
    img, _ = synthetic_foot(noise=0.5)
    assert 24 < otsu_threshold(img) < 31
    with pytest.raises(ValueError):
        otsu_threshold(np.ones((4, 4)))
    with pytest.raises(ValueError):
        otsu_threshold(np.full((4, 4), np.nan))


def test_otsu_threshold_midway_between_two_levels():
    img, _ = synthetic_foot(warm=34.0, cold=22.0)
    assert otsu_threshold(img) == pytest.approx(28.0, abs=0.2)
    img[0, :] = 22.4  # slightly warmer background rows must stay below the threshold
    assert segment_feet(img)[0][0][0, :].sum() == 0


def test_segment_single_foot_recovers_shape():
    img, truth = synthetic_foot(noise=0.5)
    masks, cut = segment_feet(img)
    assert len(masks) == 1 and 24 < cut < 31
    iou = (masks[0] & truth).sum() / (masks[0] | truth).sum()
    assert iou > 0.99


def test_segment_two_feet_sorted_left_to_right_and_noise_ignored():
    left, _ = synthetic_foot(w=400, cx=100)
    right, _ = synthetic_foot(w=400, cx=300, rx=40)
    img = np.maximum(left, right)
    img[5, 5] = 40.0  # isolated hot pixel must not count as a foot
    masks, _ = segment_feet(img, n_feet=2)
    assert len(masks) == 2
    assert np.nonzero(masks[0])[1].mean() < np.nonzero(masks[1])[1].mean()
    assert masks[0].sum() > masks[1].sum()  # wider left foot


def test_segment_holes_filled_and_nan_excluded():
    img, truth = synthetic_foot()
    img[140:160, 90:110] = 22.0  # cold spot inside the foot (e.g. a lesion/cold area)
    img[0:10, :] = np.nan  # decoded-invalid band (legend/background)
    masks, _ = segment_feet(img)
    assert masks[0][150, 100] and not masks[0][0:10].any()


def test_segment_no_foot_returns_empty():
    img = np.full((50, 50), 25.0)
    img[0, 0] = 26.0  # single pixel is below min_area_fraction
    masks, _ = segment_feet(img, threshold=25.5)
    assert masks == []
    assert segment_feet(np.full((20, 20), 25.0), threshold=30.0)[0] == []


@pytest.mark.parametrize("heel,hind_row_range", [("bottom", (200, 270)), ("top", (30, 100))])
def test_split_regions_partition_and_heel_orientation(heel, hind_row_range):
    img, truth = synthetic_foot()
    regions = split_foot_regions(truth, heel)
    assert tuple(regions) == FOOT_REGION_NAMES
    stacked = np.stack(list(regions.values()))
    assert (stacked.sum(axis=0) <= 1).all()  # disjoint
    assert np.array_equal(stacked.any(axis=0), truth)  # exact union
    rows = np.nonzero(regions["Hindfoot"])[0]
    assert hind_row_range[0] < rows.mean() < hind_row_range[1]
    # equal thirds of the long axis (foot spans ~240 rows) -> ~80 rows each
    for name in FOOT_REGION_NAMES:
        r = np.nonzero(regions[name])[0]
        assert r.max() - r.min() == pytest.approx(80, abs=4)


def test_split_regions_horizontal_foot_and_custom_fractions():
    img, truth = synthetic_foot(h=200, w=300, cx=150, cy=100, ry=45, rx=120)  # long axis horizontal
    heel_right = split_foot_regions(truth, "right", fractions=(0.4, 0.7))
    cols = {n: np.nonzero(m)[1].mean() for n, m in heel_right.items()}
    assert cols["Hindfoot"] > cols["Midfoot"] > cols["Forefoot"]
    length = 240.0
    assert heel_right["Hindfoot"].any(axis=0).sum() == pytest.approx(0.4 * length, abs=4)


def test_split_regions_tilted_foot_follows_long_axis():
    img, truth = synthetic_foot(angle=25)
    regions = split_foot_regions(truth, "bottom")
    assert np.array_equal(np.stack(list(regions.values())).any(axis=0), truth)
    areas = [m.sum() for m in regions.values()]
    # thirds of an ellipse's length: end caps are smaller than the middle section
    assert areas[1] > areas[0] and areas[1] > areas[2]


def test_split_regions_validation():
    _, truth = synthetic_foot()
    with pytest.raises(ValueError):
        split_foot_regions(truth, "sideways")
    for bad in [(0.7, 0.3), (0.0, 0.5), (0.3, 1.0)]:
        with pytest.raises(ValueError):
            split_foot_regions(truth, "bottom", bad)
    with pytest.raises(ValueError):
        split_foot_regions(np.zeros((10, 10), dtype=bool), "bottom")


def test_regions_round_trip_through_editable_canvas_polygons():
    """Auto regions -> Fabric polygons on a downscaled canvas -> masks must match."""
    img, truth = synthetic_foot(h=600, w=400, cx=200, cy=300, ry=240, rx=90, angle=12)
    regions = list(split_foot_regions(truth, "bottom").values())
    drawing = masks_to_fabric_drawing(regions, canvas_width=400, canvas_height=600,
                                      original_width=400, original_height=600)
    assert len(drawing["objects"]) == 3
    back = masks_from_canvas(drawing, canvas_width=400, canvas_height=600,
                             original_width=400, original_height=600)
    assert len(back) == 3
    for original, recovered in zip(regions, back):
        assert (original & recovered).sum() / (original | recovered).sum() > 0.97
        assert recovered.sum() == pytest.approx(original.sum(), rel=0.03)
    # also at a different canvas scale (900-wide canvas cap on big images)
    big = masks_from_canvas(
        masks_to_fabric_drawing(regions, canvas_width=200, canvas_height=300, original_width=400, original_height=600),
        canvas_width=200, canvas_height=300, original_width=400, original_height=600)
    assert big[1].sum() == pytest.approx(regions[1].sum(), rel=0.03)


def test_canvas_size_scales_small_maps_up_and_large_down():
    from roi_core import canvas_size
    assert canvas_size(80, 60) == (900, 675)  # small CSV grid is drawable
    assert canvas_size(4000, 3000) == (900, 675)
    assert canvas_size(60, 1000) == (72, 1200)  # tall maps limited by height
    assert canvas_size(1, 1) == (900, 900) and canvas_size(900, 600) == (900, 600)
    with pytest.raises(ValueError):
        canvas_size(0, 10)


def test_contour_area_exact_on_upscaled_small_csv_canvas():
    from roi_core import canvas_size
    cw, ch = canvas_size(80, 60)  # 900 x 675: factor 11.25
    obj = {"type": "rect", "left": 112.5, "top": 112.5, "width": 337.5, "height": 225}  # 30 x 20 px
    assert _area(obj, cw, ch, 80, 60) == 600


def test_strict_mode_excludes_warm_bedding_touching_foot():
    from roi_core import default_morphology_radii, strict_threshold
    img, truth = synthetic_foot(h=300, w=300, cx=100, cy=150, ry=110, rx=40, warm=33.0, cold=22.0)
    yy, xx = np.mgrid[0:300, 0:300]
    bedding = (xx > 135) & (xx < 280) & (yy > 100) & (yy < 250)  # warm blanket, 27 C, touching the foot
    img = np.where(bedding & ~truth, 27.0, img)
    plain, _ = segment_feet(img)
    assert plain[0].sum() > 1.5 * truth.sum()  # default lumps blanket + foot together
    base, core = strict_threshold(img)
    assert core > base
    opening, grow = default_morphology_radii(img.shape)
    strict, used = segment_feet(img, threshold=base, core_threshold=core, opening_radius=opening, grow_radius=grow)
    assert used == core
    iou = (strict[0] & truth).sum() / (strict[0] | truth).sum()
    assert iou > 0.9


def test_strict_threshold_falls_back_on_two_level_image():
    from roi_core import strict_threshold
    img, _ = synthetic_foot()
    base, core = strict_threshold(img)
    assert core == base


def test_opening_removes_thin_bridge_between_blobs():
    img = np.full((100, 100), 20.0)
    img[10:50, 10:50] = 30.0
    img[10:50, 60:90] = 30.0
    img[29:31, 49:61] = 30.0  # 2-px-tall bridge joining the blobs
    joined, _ = segment_feet(img, threshold=25, n_feet=2)
    assert len(joined) == 1
    cut, _ = segment_feet(img, threshold=25, n_feet=2, opening_radius=3)
    assert len(cut) == 2


def test_trim_nan_border_removes_text_header_row_and_label_column():
    from roi_core import trim_nan_border
    core = np.arange(12, dtype=float).reshape(3, 4)
    padded = np.full((5, 6), np.nan)
    padded[1:4, 1:5] = core
    assert np.array_equal(trim_nan_border(padded), core)
    interior = core.copy(); interior[1, 1] = np.nan  # interior NaN must stay in place
    assert np.isnan(trim_nan_border(interior)[1, 1]) and trim_nan_border(interior).shape == (3, 4)
    assert trim_nan_border(np.full((2, 2), np.nan)).shape == (2, 2)


def test_flir_style_csv_text_header_parses_to_clean_grid():
    import pandas as pd
    from roi_core import trim_nan_border
    text = "File:,C:\\x\\a.jpg,,\nFrame 1,20.1,20.2,20.3\n,20.4,20.5,20.6\n"
    frame = pd.read_csv(io.StringIO(text), header=None)
    grid = trim_nan_border(frame.apply(pd.to_numeric, errors="coerce").to_numpy(float))
    assert grid.shape == (2, 3) and grid[0, 0] == 20.1 and grid[1, 2] == 20.6


def test_split_hand_drawn_polygon_into_three_exact_regions():
    """Draw-then-split workflow: canvas polygon -> mask -> hind/mid/fore partition."""
    th = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    pts = np.column_stack([100 + 40 * np.cos(th), 150 + 85 * np.sin(th)])
    lo = pts.min(0)
    obj = {"type": "polygon", "originX": "left", "originY": "top", "left": lo[0], "top": lo[1],
           "width": np.ptp(pts[:, 0]), "height": np.ptp(pts[:, 1]),
           "points": [{"x": float(a), "y": float(b)} for a, b in pts - lo]}
    mask = masks_from_canvas({"objects": [obj]}, canvas_width=300, canvas_height=300,
                             original_width=300, original_height=300)[0]
    regions = split_foot_regions(mask, "bottom", (0.4, 0.7))
    stacked = np.stack(list(regions.values()))
    assert (stacked.sum(0) <= 1).all() and np.array_equal(stacked.any(0), mask)
    rows = {n: np.nonzero(m)[0].mean() for n, m in regions.items()}
    assert rows["Hindfoot"] > rows["Midfoot"] > rows["Forefoot"]  # heel at bottom of image
    length = np.ptp(np.nonzero(mask)[0]) + 1
    assert np.ptp(np.nonzero(regions["Hindfoot"])[0]) + 1 == pytest.approx(0.4 * length, abs=3)


def test_otsu_threshold_gives_near_perfect_synthetic_foot_mask():
    img, truth = synthetic_foot()
    masks, _ = segment_feet(img, threshold=otsu_threshold(img))
    assert (masks[0] & truth).sum() / (masks[0] | truth).sum() > 0.99


# ---- Shape-based foot template fit (synthetic feet only) -------------------

from roi_core import (  # noqa: E402
    fit_feet,
    fit_foot_template,
    rasterize_polygon,
    template_landmarks,
    template_polygon,
    template_regions,
)


def _synthetic_scene(params_list, shape=(300, 400), bedding=True, seed=0):
    """Warm template-shaped feet on a cool background with blur, noise and warm bedding."""
    from scipy.ndimage import gaussian_filter

    h, w = shape
    truth = [rasterize_polygon(template_polygon(p, "bottom"), h, w) for p in params_list]
    img = np.full(shape, 22.0)
    for m in truth:
        img[m] = 33.0
    if bedding:
        yy, xx = np.mgrid[0:h, 0:w]
        blob = ((xx - 200) ** 2 / 60**2 + (yy - 170) ** 2 / 50**2) <= 1  # 29 C fabric between the feet
        img[blob & ~np.any(truth, axis=0)] = 29.0
    img = gaussian_filter(img, 1.5) + np.random.default_rng(seed).normal(0, 0.15, shape)
    return img, truth


def test_template_landmarks_are_inside_the_foot_and_ordered():
    arch_start, ball = template_landmarks()
    assert 0.2 < arch_start < 0.4 and 0.6 < ball < 0.85 and arch_start < ball


def test_template_polygon_is_foot_shaped():
    p = {"cx": 100, "cy": 150, "length": 200, "theta": 0, "aspect": 1.0, "mirror": 1}
    poly = template_polygon(p, "bottom")
    ys = poly[:, 1]
    assert ys.max() - ys.min() == pytest.approx(200, abs=1)  # length
    width_at = lambda y0: np.ptp(poly[np.abs(ys - y0) < 4][:, 0])  # noqa: E731
    toe_y, heel_y = ys.min(), ys.max()
    ball_w = width_at(heel_y - 0.72 * 200)
    arch_w = width_at(heel_y - 0.42 * 200)
    heel_w = width_at(heel_y - 0.15 * 200)
    assert ball_w > heel_w > arch_w * 0.95  # ball widest, arch narrowest region
    assert ball_w == pytest.approx(0.38 * 200, rel=0.1)
    # mirror flips the shape left-right about the axis
    flipped = template_polygon({**p, "mirror": -1}, "bottom")
    assert np.allclose(np.sort(flipped[:, 0] - 100), np.sort(-(poly[:, 0] - 100)), atol=1e-6)


def test_template_regions_partition_the_template():
    p = {"cx": 100, "cy": 150, "length": 200, "theta": 15, "aspect": 1.0, "mirror": 1}
    regions = template_regions(p, "bottom", (300, 200))
    stacked = np.stack(list(regions.values()))
    outline = rasterize_polygon(template_polygon(p, "bottom"), 300, 200)
    assert (stacked.sum(0) <= 1).all() and np.array_equal(stacked.any(0), outline)
    assert regions["Hindfoot"].sum() > 0 and regions["Midfoot"].sum() > regions["Hindfoot"].sum() * 0.5
    arch_start, ball = template_landmarks()
    custom = template_regions(p, "bottom", (300, 200), boundaries=(0.5, 0.9))
    assert custom["Hindfoot"].sum() > regions["Hindfoot"].sum()


@pytest.mark.parametrize("theta,mirror", [(0, 1), (12, -1), (-15, 1)])
def test_fit_recovers_single_foot_despite_bedding(theta, mirror):
    truth_params = {"cx": 120, "cy": 150, "length": 200, "theta": theta, "aspect": 1.0, "arch": 1.0, "mirror": mirror}
    img, (truth,) = _synthetic_scene([truth_params], shape=(300, 300), bedding=False)
    yy, xx = np.mgrid[0:300, 0:300]
    bedding = ((xx - 215) ** 2 / 45**2 + (yy - 170) ** 2 / 45**2) <= 1
    img = np.where(bedding & ~truth, 29.0, img)
    # rough seed: the foot with its cooler toe end and heel end cut off + shifted
    ys, xs = np.nonzero(truth)
    seed = truth & (yy > np.percentile(ys, 18)) & (yy < np.percentile(ys, 92))
    fit = fit_foot_template(img, "bottom", seed)
    mask = rasterize_polygon(fit["polygon"], 300, 300)
    iou = (mask & truth).sum() / (mask | truth).sum()
    assert iou > 0.85
    assert fit["mirror"] == mirror


def test_fit_feet_pair_gets_opposite_mirrors_and_both_feet_found():
    left = {"cx": 110, "cy": 150, "length": 200, "theta": -8, "aspect": 1.0, "arch": 1.0, "mirror": 1}
    right = {"cx": 290, "cy": 150, "length": 200, "theta": 8, "aspect": 1.0, "arch": 1.0, "mirror": -1}
    img, truth = _synthetic_scene([left, right], shape=(300, 400))
    seeds, _ = segment_feet(img, n_feet=2)
    fits = fit_feet(img, "bottom", seeds)
    assert [f["mirror"] for f in fits] == [1, -1]
    for fit, t in zip(fits, truth):
        mask = rasterize_polygon(fit["polygon"], 300, 400)
        assert (mask & t).sum() / (mask | t).sum() > 0.85


def test_fit_validation_errors():
    img, (truth,) = _synthetic_scene([{"cx": 100, "cy": 150, "length": 200, "theta": 0, "aspect": 1.0, "mirror": 1}],
                                     shape=(300, 200), bedding=False)
    with pytest.raises(ValueError):
        fit_foot_template(img, "sideways", truth)
    with pytest.raises(ValueError):
        fit_foot_template(img, "bottom", np.zeros_like(truth))
    with pytest.raises(ValueError):
        fit_foot_template(img, "bottom", truth[:10])


# ---- Fabric.js object placement (regression: circle tool rotates about the drag start) ----


def _circle_json(x0, y0, x1, y1):
    """What the canvas circle tool sends for a drag from (x0, y0) to (x1, y1):
    diameter = drag length, origin = start point (left/center), angle = drag direction."""
    import math

    diameter = math.hypot(x1 - x0, y1 - y0)
    return {"type": "circle", "originX": "left", "originY": "center", "left": x0, "top": y0,
            "width": diameter, "height": diameter, "radius": diameter / 2,
            "angle": math.degrees(math.atan2(y1 - y0, x1 - x0)), "scaleX": 1, "scaleY": 1}


@pytest.mark.parametrize("drag", [(100, 100, 180, 160), (250, 200, 150, 120), (100, 220, 190, 150),
                                  (260, 90, 200, 210), (150, 150, 250, 150), (150, 150, 150, 250)])
def test_circle_tool_centre_is_midpoint_of_drag_in_every_direction(drag):
    x0, y0, x1, y1 = drag
    mask = masks_from_canvas({"objects": [_circle_json(*drag)]}, canvas_width=400, canvas_height=300,
                             original_width=400, original_height=300)[0]
    ys, xs = np.nonzero(mask)
    assert xs.mean() + 0.5 == pytest.approx((x0 + x1) / 2, abs=0.6)
    assert ys.mean() + 0.5 == pytest.approx((y0 + y1) / 2, abs=0.6)
    radius = np.hypot(x1 - x0, y1 - y0) / 2
    assert mask.sum() == pytest.approx(np.pi * radius**2, rel=0.01)


def test_exact_circle_json_captured_from_the_canvas():
    obj = {"type": "circle", "originX": "left", "originY": "center", "left": 99, "top": 99, "width": 100,
           "height": 100, "scaleX": 1, "scaleY": 1, "angle": 36.87, "radius": 50}
    ys, xs = np.nonzero(masks_from_canvas({"objects": [obj]}, canvas_width=400, canvas_height=300,
                                          original_width=400, original_height=300)[0])
    assert (xs.mean() + 0.5, ys.mean() + 0.5) == pytest.approx((139, 129), abs=0.6)  # not (149, 99)


def test_circle_position_scales_with_canvas_to_source_ratio():
    obj = _circle_json(100, 100, 180, 160)
    mask = masks_from_canvas({"objects": [obj]}, canvas_width=400, canvas_height=300,
                             original_width=800, original_height=600)[0]
    ys, xs = np.nonzero(mask)
    assert (xs.mean() + 0.5, ys.mean() + 0.5) == pytest.approx((280, 260), abs=1.2)


def test_rotated_rect_with_corner_origin_pivots_about_its_origin_point():
    # origin top-left at (100, 100), 100 x 40, rotated 90 degrees clockwise about that corner
    obj = {"type": "rect", "originX": "left", "originY": "top", "left": 100, "top": 100,
           "width": 100, "height": 40, "angle": 90, "scaleX": 1, "scaleY": 1}
    ys, xs = np.nonzero(masks_from_canvas({"objects": [obj]}, canvas_width=300, canvas_height=300,
                                          original_width=300, original_height=300)[0])
    assert (xs.min(), xs.max() + 1, ys.min(), ys.max() + 1) == (60, 100, 100, 200)


# ---- Left-right comparison -------------------------------------------------

from roi_core import default_region_pairs, left_right_comparison  # noqa: E402


def _two_roi_scene():
    values = np.full((20, 40), 25.0)
    values[:, :20] = 33.0  # left half warmer
    values[:, 20:] = 30.5
    a = np.zeros((20, 40), dtype=bool); a[2:12, 2:12] = True    # 100 px at 33.0
    b = np.zeros((20, 40), dtype=bool); b[2:12, 22:37] = True   # 150 px at 30.5
    return values, [a, b]


def test_left_right_differences_are_a_minus_b():
    values, masks = _two_roi_scene()
    (row,) = left_right_comparison(values, masks, [("Whole", 1, 2)], row_spacing=2.0, column_spacing=0.5)
    assert row["mean_a"] == 33.0 and row["mean_b"] == 30.5
    assert row["mean_difference"] == pytest.approx(2.5) and row["median_difference"] == pytest.approx(2.5)
    assert row["p95_difference"] == pytest.approx(2.5)
    assert (row["area_a"], row["area_b"]) == (100.0, 150.0)  # pixels x 2.0 x 0.5
    assert row["area_difference"] == -50.0
    assert row["area_difference_percent"] == pytest.approx(100 * -50 / 125)  # symmetric %
    assert "exceeds_threshold" not in row
    (swapped,) = left_right_comparison(values, masks, [("Whole", 2, 1)])
    assert swapped["mean_difference"] == pytest.approx(-2.5)


def test_left_right_threshold_flag_is_optional_and_inclusive():
    values, masks = _two_roi_scene()
    assert left_right_comparison(values, masks, [("x", 1, 2)], threshold=2.5)[0]["exceeds_threshold"] is True
    assert left_right_comparison(values, masks, [("x", 1, 2)], threshold=2.6)[0]["exceeds_threshold"] is False
    assert left_right_comparison(values, masks, [("x", 2, 1)], threshold=2.5)[0]["exceeds_threshold"] is True  # |diff|


def test_left_right_handles_nan_and_empty_valid_pixels():
    values, masks = _two_roi_scene()
    values = values.copy(); values[masks[1]] = np.nan
    row = left_right_comparison(values, masks, [("x", 1, 2)], threshold=1.0)[0]
    assert row["valid_pixels_b"] == 0 and np.isnan(row["mean_b"]) and np.isnan(row["mean_difference"])
    assert row["exceeds_threshold"] is False
    assert row["area_b"] == 150.0  # geometry unaffected by missing values


def test_left_right_validation():
    values, masks = _two_roi_scene()
    with pytest.raises(ValueError):
        left_right_comparison(values, masks, [("x", 1, 3)])
    with pytest.raises(ValueError):
        left_right_comparison(values, masks, [("x", 0, 2)])
    with pytest.raises(ValueError):
        left_right_comparison(values, masks, [("x", 2, 2)])
    assert left_right_comparison(values, masks, []) == []


def test_default_pairs_from_foot_labels_and_fallbacks():
    labels = ["Foot 1 hindfoot", "Foot 1 midfoot", "Foot 1 forefoot",
              "Foot 2 hindfoot", "Foot 2 midfoot", "Foot 2 forefoot"]
    assert default_region_pairs(labels, 6) == [("Hindfoot", 1, 4), ("Midfoot", 2, 5), ("Forefoot", 3, 6)]
    contour_labels = [f"Contour {c} {r}" for c in (1, 2) for r in ("hindfoot", "midfoot", "forefoot")]
    assert default_region_pairs(contour_labels, 6) == [("Hindfoot", 1, 4), ("Midfoot", 2, 5), ("Forefoot", 3, 6)]
    assert default_region_pairs(labels, 5) == [("Pair 1", 1, 2)]  # contours changed: labels no longer apply
    assert default_region_pairs(None, 2) == [("Pair 1", 1, 2)]
    assert default_region_pairs(None, 1) == [] and default_region_pairs(labels[:3], 1) == []
    one_foot = labels[:3]
    assert default_region_pairs(one_foot, 3) == [("Pair 1", 1, 2)]  # only one foot: no automatic pairing
