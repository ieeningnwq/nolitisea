"""Tests for nolitisea.utils.resample."""
import unittest

import numpy as np

from nolitisea.utils.resample import resample


class TestResample(unittest.TestCase):
    def setUp(self):
        # Sine signal: period = 20 samples
        self.n = 200
        self.t = np.arange(self.n, dtype=float)
        self.x = np.sin(2 * np.pi * self.t / 20.0)

    # ---------- basic functionality ----------

    def test_upsample(self):
        """Up-sample with sampletime=0.5 -> roughly twice as many output points."""
        y = resample(self.x, sampletime=0.5, order=4)
        # start = (horder+1)/2 = 3.0, end = n - horder//2 = 198
        expected_len = len(np.arange(3.0, 198, 0.5))
        self.assertEqual(y.size, expected_len)
        # Compare with the true sine; error should be small
        times = np.arange(3.0, 198, 0.5)
        expected = np.sin(2 * np.pi * times / 20.0)
        np.testing.assert_allclose(y, expected, atol=1e-3)

    def test_downsample(self):
        """Down-sample with sampletime=2.0 -> roughly half as many output points."""
        y = resample(self.x, sampletime=2.0, order=4)
        expected_len = len(np.arange(3.0, 198, 2.0))
        self.assertEqual(y.size, expected_len)
        times = np.arange(3.0, 198, 2.0)
        expected = np.sin(2 * np.pi * times / 20.0)
        np.testing.assert_allclose(y, expected, atol=1e-3)

    def test_identity_rate(self):
        """sampletime=1.0 -> same sampling rate, integer points recovered exactly."""
        y = resample(self.x, sampletime=1.0, order=4)
        times = np.arange(3.0, 198, 1.0)
        expected = np.sin(2 * np.pi * times / 20.0)
        np.testing.assert_allclose(y, expected, atol=1e-10)

    def test_linear_order1_extrapolation(self):
        """order=1 output times are all half-integers, equivalent to linear extrapolation.

        horder=2, horder2=-1, start=1.5. For each output time t=k+0.5:
          itime = k-1, htime = 0.5, nodes (-1, series[k-1]), (0, series[k])
          p(0.5) = 1.5*series[k] - 0.5*series[k-1]
        """
        y = resample(self.x, sampletime=1.0, order=1)
        for k_idx, time in enumerate(np.arange(1.5, self.n - 1, 1.0)):
            i = int(time)  # floor(1.5)=1, floor(2.5)=2, ...
            expected = 1.5 * self.x[i] - 0.5 * self.x[i - 1]
            self.assertAlmostEqual(y[k_idx], expected, places=10)

    def test_constant_series(self):
        """A constant series remains constant after resampling."""
        xc = np.ones(100)
        for st in (0.5, 1.0, 2.0):
            yc = resample(xc, sampletime=st, order=4)
            np.testing.assert_allclose(yc, 1.0)

    def test_polynomial_exact(self):
        """A degree-order polynomial is interpolated exactly at integer points."""
        # 4th-degree polynomial + order=4 -> exact recovery
        x_poly = (self.t ** 4 - 3 * self.t ** 3 + 2 * self.t ** 2 - self.t) * 1e-6
        y = resample(x_poly, sampletime=1.0, order=4)
        times = np.arange(3.0, self.n - 2, 1.0)
        expected = (times ** 4 - 3 * times ** 3 + 2 * times ** 2 - times) * 1e-6
        np.testing.assert_allclose(y, expected, atol=1e-8)

    # ---------- parameter validation ----------

    def test_invalid_order(self):
        with self.assertRaises(ValueError):
            resample(self.x, order=0)

    def test_invalid_sampletime(self):
        with self.assertRaises(ValueError):
            resample(self.x, sampletime=0.0)
        with self.assertRaises(ValueError):
            resample(self.x, sampletime=-1.0)

    def test_series_too_short(self):
        with self.assertRaises(ValueError):
            resample(np.array([1.0, 2.0]), order=4)

    def test_higher_order_needs_more_data(self):
        """order=10 requires 11 samples; a short series should raise."""
        with self.assertRaises(ValueError):
            resample(np.arange(8.0), order=10)

    # ---------- output range ----------

    def test_output_length_upsample(self):
        """Up-sampling produces a longer output."""
        y = resample(self.x, sampletime=0.25, order=4)
        self.assertGreater(y.size, self.n)

    def test_output_length_downsample(self):
        """Down-sampling produces a shorter output."""
        y = resample(self.x, sampletime=4.0, order=4)
        self.assertLess(y.size, self.n)

    def test_order_comparison(self):
        """Higher order gives better accuracy near the edges."""
        # Compare order=1 and order=4 on the sine signal
        y1 = resample(self.x, sampletime=0.5, order=1)
        y4 = resample(self.x, sampletime=0.5, order=4)
        times1 = np.arange(1.5, self.n - 1, 0.5)
        times4 = np.arange(3.0, self.n - 2, 0.5)
        exp1 = np.sin(2 * np.pi * times1 / 20.0)
        exp4 = np.sin(2 * np.pi * times4 / 20.0)
        err1 = np.max(np.abs(y1 - exp1))
        err4 = np.max(np.abs(y4 - exp4))
        self.assertLess(err4, err1)


if __name__ == "__main__":
    unittest.main()
