import unittest

import numpy as np

from diffaxisem3d.plotting import model_figure
from matplotlib import pyplot as plt


class RecoveryPlotTests(unittest.TestCase):
    def test_sections_share_the_full_in_domain_range(self):
        coordinates = np.arange(-2, 3) * 1000.0
        shape = (5, 5, 5)
        mask = np.ones(shape, dtype=bool)
        mask[0, 0, :] = False
        data = dict(x=coordinates, y=coordinates, depth=np.arange(5) * 250.0, display_mask=mask)
        for key in ("true", "initial", "inverted"):
            data[key] = np.full(shape, 2000.0)
        data["true"][~mask] = 9000.0
        data["initial"][1, 1, 0] = 1000.0
        data["inverted"][3, 3, 4] = 6500.0
        acquisition = {
            "receiver_grid": {"x_m": [0.0], "y_m": [0.0]},
            "sources": [{"xyz_depth_m": [0.0, 0.0, 0.0]}],
        }

        figure = model_figure(data, acquisition)
        self.addCleanup(plt.close, figure)
        figure.canvas.draw()
        images = [image for ax in figure.axes for image in ax.images]
        self.assertEqual(len(images), 9)
        for image in images:
            self.assertEqual(image.get_clim(), (1.0, 6.5))
            np.testing.assert_array_equal(np.unique(image.get_array().compressed()), [2.0])
            transform = image.axes.transData.get_affine().get_matrix()
            self.assertAlmostEqual(abs(transform[0, 0]), abs(transform[1, 1]))
