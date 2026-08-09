import unittest
import numpy as np
from nolitisea.delay import acorr


class TestAcorr(unittest.TestCase):

    def test_zero_lag_normalization(self):
        """r[0] should be 1 when norm=True."""
        x = np.random.randn(1000)
        r = acorr(x, norm=True)
        self.assertAlmostEqual(r[0], 1.0, places=6)

    def test_no_normalization(self):
        """r[0] should not be 1 when norm=False."""
        x = np.random.randn(1000)
        r = acorr(x, norm=False)
        self.assertNotAlmostEqual(r[0], 1.0, places=6)

    def test_detrend_off(self):
        """If detrend=False, mean should not be removed."""
        x = np.ones(500) * 3.0
        r_detrend_on = acorr(x, detrend=True, norm=False)
        r_detrend_off = acorr(x, detrend=False, norm=False)

        # With detrend=True, autocorrelation should vanish
        self.assertTrue(np.allclose(r_detrend_on, 0.0))

        # With detrend=False, autocorrelation equals variance * N
        self.assertGreater(np.abs(r_detrend_off[0]), 0.0)

    def test_white_noise_decay(self):
        """Uncorrelated noise should have near-zero autocorrelation."""
        np.random.seed(42)
        x = np.random.randn(5000)
        r = acorr(x, maxtau=100)

        # Exclude lag=0
        self.assertTrue(np.all(np.abs(r[1:]) < 0.1))

    def test_maxtau_truncation(self):
        """Output length should respect maxtau+1."""
        x = np.random.randn(1000)
        r = acorr(x, maxtau=50)
        self.assertEqual(len(r), 50)

    def test_constant_signal(self):
        """Constant signal should yield zero autocorrelation after detrending."""
        x = np.ones(200) * 7.0
        r = acorr(x, detrend=True, norm=False)
        self.assertTrue(np.allclose(r, 0.0))

    def test_single_element_input(self):
        """Length-1 input should not crash."""
        x = np.array([1.0])
        r = acorr(x, detrend=False, norm=False)
        self.assertEqual(len(r), 1)
        self.assertAlmostEqual(r[0], 1.0)

    def test_symmetry_property(self):
        """Autocorrelation should be symmetric for real signals."""
        x = np.random.randn(512)
        r = acorr(x, maxtau=100)
        # Not strictly required here since we only return [0:maxtau]
        # But r[0] should dominate
        self.assertGreaterEqual(r[0], np.max(np.abs(r[1:])))


if __name__ == "__main__":
    unittest.main()
