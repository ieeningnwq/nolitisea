import unittest

import numpy as np

from nolitisea.core.embed import delay_vectors


class TestDelayVectors(unittest.TestCase):
    """Unit tests for delay_vectors."""

    # ------------------------------------------------------------------
    # Happy-path tests
    # ------------------------------------------------------------------

    def test_univariate_default_embdim_2(self):
        """C defaults: embdim=2, delay=1; vectors anchored at the newest sample."""
        series = np.arange(5.0)
        result = delay_vectors(series)
        expected = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0]])
        np.testing.assert_array_equal(result, expected)

    def test_univariate_embdim_3_delay_2(self):
        """Offsets 0, 2, 4; base time n = r + 4."""
        series = np.arange(10.0)
        result = delay_vectors(series, embdim=3, delay=2)
        self.assertEqual(result.shape, (6, 3))
        np.testing.assert_array_equal(result[:, 0], series[4:10])
        np.testing.assert_array_equal(result[:, 1], series[2:8])
        np.testing.assert_array_equal(result[:, 2], series[0:6])

    def test_bivariate_uniform_split(self):
        """embdim split evenly across variables (C: -m without -F)."""
        data = np.vstack([np.arange(10.0), np.arange(100.0, 110.0)])
        result = delay_vectors(data, embdim=4, delay=2)
        self.assertEqual(result.shape, (8, 4))
        np.testing.assert_array_equal(result[:, 0], data[0, 2:10])
        np.testing.assert_array_equal(result[:, 1], data[0, 0:8])
        np.testing.assert_array_equal(result[:, 2], data[1, 2:10])
        np.testing.assert_array_equal(result[:, 3], data[1, 0:8])

    def test_dims_per_variable(self):
        """Per-variable dims (C: -F); embdim defaults to sum(dims)."""
        data = np.vstack([np.arange(10.0), np.arange(100.0, 110.0)])
        result = delay_vectors(data, delay=1, dims=[3, 1])
        self.assertEqual(result.shape, (8, 4))
        np.testing.assert_array_equal(result[:, 0], data[0, 2:10])
        np.testing.assert_array_equal(result[:, 1], data[0, 1:9])
        np.testing.assert_array_equal(result[:, 2], data[0, 0:8])
        np.testing.assert_array_equal(result[:, 3], data[1, 2:10])

    def test_dims_with_matching_embdim(self):
        """dims and embdim consistent (C: -F with -m)."""
        series = np.arange(8.0)
        result = delay_vectors(series, embdim=4, dims=[4])
        # offsets [0, 1, 2, 3]; n_points = 8 - 3 = 5
        self.assertEqual(result.shape, (5, 4))
        np.testing.assert_array_equal(result[:, 0], series[3:8])
        np.testing.assert_array_equal(result[:, 1], series[2:7])
        np.testing.assert_array_equal(result[:, 2], series[1:6])
        np.testing.assert_array_equal(result[:, 3], series[0:5])

    def test_increments_single_variable(self):
        """Per-coordinate increments (C: -D); consumed across variables."""
        series = np.arange(8.0)
        result = delay_vectors(series, dims=[3], increments=[1, 3])
        self.assertEqual(result.shape, (4, 3))
        np.testing.assert_array_equal(result[:, 0], series[4:8])
        np.testing.assert_array_equal(result[:, 1], series[3:7])
        np.testing.assert_array_equal(result[:, 2], series[0:4])

    def test_increments_across_variables(self):
        """Increments are consumed in order across variables."""
        data = np.vstack([np.arange(10.0), np.arange(100.0, 110.0)])
        result = delay_vectors(data, dims=[2, 2], increments=[1, 3])
        # offsets: var0 -> [0, 1], var1 -> [0, 3]; max offset = 3
        self.assertEqual(result.shape, (7, 4))
        np.testing.assert_array_equal(result[:, 0], data[0, 3:10])
        np.testing.assert_array_equal(result[:, 1], data[0, 2:9])
        np.testing.assert_array_equal(result[:, 2], data[1, 3:10])
        np.testing.assert_array_equal(result[:, 3], data[1, 0:7])

    def test_accepts_list_input(self):
        """Plain lists should be accepted like numpy arrays."""
        result = delay_vectors([1, 2, 3, 4])
        expected = np.array([[2, 1], [3, 2], [4, 3]])
        np.testing.assert_array_equal(result, expected)

    def test_output_dtype_matches_input(self):
        """Output dtype should match input dtype."""
        series = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        result = delay_vectors(series)
        self.assertEqual(result.dtype, np.int32)

    # ------------------------------------------------------------------
    # Error-condition tests
    # ------------------------------------------------------------------

    def test_raises_on_embdim_not_divisible(self):
        """C: 'Inconsistent -m and -M. Please set -F'."""
        data = np.vstack([np.arange(10.0), np.arange(10.0)])
        with self.assertRaises(ValueError) as cm:
            delay_vectors(data, embdim=3)
        self.assertIn("multiple", str(cm.exception))

    def test_raises_on_dims_length_mismatch(self):
        """dims entries must match the number of variables."""
        data = np.vstack([np.arange(10.0), np.arange(10.0)])
        with self.assertRaises(ValueError) as cm:
            delay_vectors(data, dims=[2])
        self.assertIn("variable(s)", str(cm.exception))

    def test_raises_on_dims_embdim_mismatch(self):
        """embdim must equal sum(dims) when both are given."""
        series = np.arange(10.0)
        with self.assertRaises(ValueError) as cm:
            delay_vectors(series, embdim=5, dims=[2])
        self.assertIn("does not match", str(cm.exception))

    def test_raises_on_dims_entry_below_one(self):
        """Every per-variable dimension must be >= 1."""
        series = np.arange(10.0)
        with self.assertRaises(ValueError) as cm:
            delay_vectors(series, dims=[0])
        self.assertIn(">= 1", str(cm.exception))

    def test_raises_on_delay_below_one(self):
        """delay must be >= 1."""
        series = np.arange(10.0)
        with self.assertRaises(ValueError):
            delay_vectors(series, delay=0)

    def test_raises_on_wrong_increment_count(self):
        """increments must contain embdim - n_vars entries."""
        series = np.arange(10.0)
        with self.assertRaises(ValueError) as cm:
            delay_vectors(series, embdim=3, dims=[3], increments=[1])
        self.assertIn("2 entries", str(cm.exception))

    def test_raises_on_increment_below_one(self):
        """Every increment must be >= 1."""
        series = np.arange(10.0)
        with self.assertRaises(ValueError) as cm:
            delay_vectors(series, embdim=3, dims=[3], increments=[0, 1])
        self.assertIn(">= 1", str(cm.exception))

    def test_raises_on_scalar_dims(self):
        """dims must be a sequence, not a scalar."""
        series = np.arange(10.0)
        with self.assertRaises(ValueError) as cm:
            delay_vectors(series, dims=3)
        self.assertIn("sequence", str(cm.exception))

    def test_raises_on_too_short_series(self):
        """Series shorter than the largest offset should raise ValueError."""
        series = np.arange(3.0)
        with self.assertRaises(ValueError) as cm:
            delay_vectors(series, embdim=5)
        self.assertIn("too short", str(cm.exception))

    def test_raises_on_3d_input(self):
        """Only 1-D and 2-D inputs are supported."""
        series = np.zeros((2, 2, 2))
        with self.assertRaises(ValueError):
            delay_vectors(series)


if __name__ == "__main__":
    unittest.main()
