import os
import unittest

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nolitisea.data import henon_map
from nolitisea.utils import add_noise
from nolitisea.plots import plot_clean_vs_noisy, set_publication_style


class TestPlots(unittest.TestCase):

    def setUp(self):
        set_publication_style()
        self.tmp_path = os.path.normpath("test_clean_vs_noisy.png")

    def test_plot_clean_vs_noisy(self):
        """Test that plot_clean_vs_noisy generates a figure."""
        X, _ = henon_map(a=1.4, b=0.3, transient=2000, n=5000)
        X_noisy = add_noise(X, noise_type="gaussian", level=0.05,
                            absolute=False, seed=42)

        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        plot_clean_vs_noisy(
            X, X_noisy,
            sample=200,
            xlabel="Sample index $n$",
            ylabel="$x_n$",
            title="Original vs. noisy data",
            ax=ax,
        )
        fig.savefig(self.tmp_path, dpi=300)

        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(self.tmp_path))
        self.assertGreater(os.path.getsize(self.tmp_path), 0)

if __name__ == "__main__":
    unittest.main()
