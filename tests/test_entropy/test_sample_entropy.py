import unittest

import numpy as np

from nolitisea.entropy.sample_entropy import sample_entropy


class TestSampleEntropy(unittest.TestCase):
    """Unit test class for validating sample entropy algorithm performance."""

    def setUp(self):
        """Initialize multiple standard time series datasets with fixed random seed."""
        np.random.seed(1024)
        # Constant stationary sequence with zero fluctuation
        self.constant_seq = np.full(1000, 50.0)
        # Low-fluctuation smooth sequence
        self.smooth_seq = np.random.normal(loc=50, scale=0.5, size=1000)
        # High-fluctuation noisy sequence
        self.noise_seq = np.random.normal(loc=50, scale=3.0, size=1000)
        # Short-length sequence for boundary test
        self.short_seq = np.random.randn(50)
        # Wind ramp sequence: deterministic trend + noise (strong regularity, low entropy)
        self.trend_seq = np.linspace(10, 90, 1000) + np.random.randn(1000) * 0.8

    def test_output_type_and_range(self):
        """Verify output data type and non-negative value range."""
        se_const = sample_entropy(self.constant_seq)
        se_smooth = sample_entropy(self.smooth_seq)
        se_noise = sample_entropy(self.noise_seq)

        self.assertIsInstance(se_const, float)
        self.assertIsInstance(se_smooth, float)
        self.assertIsInstance(se_noise, float)

        self.assertGreaterEqual(se_const, 0.0)
        self.assertGreaterEqual(se_smooth, 0.0)
        self.assertGreaterEqual(se_noise, 0.0)

    def test_physical_consistency(self):
        """Verify physical rationality: stationary sequence entropy < noisy sequence entropy."""
        se_smooth = sample_entropy(self.smooth_seq)
        se_noise = sample_entropy(self.noise_seq)
        self.assertLess(se_smooth, se_noise)

    def test_short_sequence_adapt(self):
        """Verify algorithm adaptability for short time series."""
        se_short = sample_entropy(self.short_seq)
        self.assertIsInstance(se_short, float)
        self.assertFalse(np.isnan(se_short))

    def test_different_param_config(self):
        """Verify algorithm stability under different embedding dimensions and thresholds."""
        se_m3 = sample_entropy(self.smooth_seq, m=3)
        se_r3 = sample_entropy(self.smooth_seq, r_multiplier=0.3)
        self.assertIsInstance(se_m3, float)
        self.assertIsInstance(se_r3, float)

    def test_trend_sequence(self):
        """Verify entropy rationality: deterministic trend sequence has lower entropy than pure noise sequence."""
        se_trend = sample_entropy(self.trend_seq)
        se_noise = sample_entropy(self.noise_seq)
        self.assertLess(se_trend, se_noise)

    def test_zero_std_sequence(self):
        """Verify extreme condition handling for zero-variance constant sequence."""
        se = sample_entropy(self.constant_seq)
        self.assertEqual(se, 0.0)



if __name__ == '__main__':
    unittest.main(verbosity=2)

