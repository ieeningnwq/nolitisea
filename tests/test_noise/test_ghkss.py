"""Tests for the GHKSS multivariate noise reduction."""

import unittest

import numpy as np

from nolitisea.generate.lorenz import lorenz
from nolitisea.noise.ghkss import ghkss


class TestGhkss(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _, states = lorenz(length=700, step=0.001, sample=0.03, discard=3000)
        rng = np.random.default_rng(42)
        cls.clean = states[:, 0].copy()
        cls.noisy = cls.clean + rng.normal(
            0.0, 0.05 * cls.clean.std(), cls.clean.shape
        )

    def test_denoises_univariate(self):
        out = ghkss(self.noisy, embed=6, delay=1, qdim=3, minn=60,
                    iterations=3)
        corrected = out["corrected"]
        self.assertEqual(corrected.shape, self.noisy.shape)
        rms_noisy = np.sqrt(np.mean((self.noisy - self.clean) ** 2))
        rms_denoised = np.sqrt(np.mean((corrected - self.clean) ** 2))
        self.assertLess(rms_denoised, 0.7 * rms_noisy)

    def test_deterministic(self):
        a = ghkss(self.noisy, embed=5, delay=2, qdim=2, minn=30)["corrected"]
        b = ghkss(self.noisy, embed=5, delay=2, qdim=2, minn=30)["corrected"]
        np.testing.assert_array_equal(a, b)

    def test_multivariate_shape_and_stats(self):
        data = np.column_stack([self.noisy, np.roll(self.noisy, 3)])
        out = ghkss(data, embed=4, delay=2, qdim=3, minn=25, iterations=2)
        self.assertEqual(out["corrected"].shape, data.shape)
        self.assertEqual(len(out["stats"]), 2)
        for entry in out["stats"]:
            self.assertEqual(len(entry["average_shift"]), 2)
            self.assertEqual(len(entry["rms_correction"]), 2)
        self.assertTrue(np.all(np.isfinite(out["corrected"])))

    def test_min_eps_absolute(self):
        interval = self.noisy.max() - self.noisy.min()
        out = ghkss(self.noisy, embed=5, delay=2, qdim=2, minn=30,
                    mineps=interval / 1000.0)
        self.assertEqual(out["corrected"].shape, self.noisy.shape)
        self.assertTrue(np.all(np.isfinite(out["corrected"])))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            ghkss(self.noisy, embed=5, delay=2, qdim=10)  # qdim >= dim
        with self.assertRaises(ValueError):
            ghkss(self.noisy[:10], embed=5, delay=2, minn=30)
        with self.assertRaises(ValueError):
            ghkss(self.noisy, embed=5, delay=2, minn=100000)
        with self.assertRaises(RuntimeError):
            ghkss(np.ones(200))  # constant series has zero range


if __name__ == "__main__":
    unittest.main()
