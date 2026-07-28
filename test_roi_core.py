import unittest

import numpy as np

from roi_core import (
    decode_false_color,
    masks_from_canvas,
    roi_geometry,
    summary_statistics,
)


class RoiCoreTests(unittest.TestCase):
    def test_summary_statistics_respects_mask_and_nan(self):
        values = np.array([[1.0, 2.0], [3.0, np.nan]])
        mask = np.array([[True, True], [False, True]])
        result = summary_statistics(values, mask)
        self.assertEqual(result["selected_pixels"], 3)
        self.assertEqual(result["valid_pixels"], 2)
        self.assertAlmostEqual(result["mean"], 1.5)

    def test_geometry_for_two_by_two_square(self):
        mask = np.ones((2, 2), dtype=bool)
        result = roi_geometry(mask, row_spacing=2.0, column_spacing=3.0)
        self.assertEqual(result["roi_pixels"], 4)
        self.assertAlmostEqual(result["area"], 24.0)
        self.assertAlmostEqual(result["perimeter"], 20.0)

    def test_canvas_rectangle_scales_to_source(self):
        canvas = {
            "objects": [
                {
                    "type": "rect",
                    "left": 2,
                    "top": 2,
                    "width": 4,
                    "height": 4,
                    "scaleX": 1,
                    "scaleY": 1,
                }
            ]
        }
        masks = masks_from_canvas(
            canvas,
            canvas_width=10,
            canvas_height=10,
            original_width=20,
            original_height=20,
        )
        self.assertEqual(len(masks), 1)
        self.assertGreater(masks[0].sum(), 0)
        self.assertEqual(masks[0].shape, (20, 20))

    def test_colormap_round_trip(self):
        from matplotlib import colormaps

        positions = np.linspace(0, 1, 20)
        rgb = colormaps["turbo"](positions, bytes=True)[..., :3]
        rgb = np.tile(rgb[np.newaxis, :, :], (4, 1, 1))
        decoded, valid, _ = decode_false_color(rgb, "turbo", 10, 20, max_rgb_distance=10)
        self.assertTrue(valid.all())
        self.assertAlmostEqual(float(np.nanmin(decoded)), 10.0, delta=0.2)
        self.assertAlmostEqual(float(np.nanmax(decoded)), 20.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()

