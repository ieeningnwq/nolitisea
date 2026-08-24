import unittest

import numpy as np

from nolitisea.core.embed import delay_embedding  # replace with actual module name


class TestDelayEmbedding(unittest.TestCase):
    """Unit tests for delay_embedding function."""

    # ------------------------------------------------------------------
    # Happy-path tests
    # ------------------------------------------------------------------

    def test_basic_embedding_dim_2_delay_1(self):
        """Standard case: dim=2, delay=1."""
        series = [1, 2, 3, 4, 5]
        result = delay_embedding(series, dim=2, delay=1)
        expected = np.array([
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
        ])
        np.testing.assert_array_equal(result, expected)

    def test_basic_embedding_dim_3_delay_1(self):
        """Standard case: dim=3, delay=1."""
        series = [1, 2, 3, 4, 5]
        result = delay_embedding(series, dim=3, delay=1)
        expected = np.array([
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
        ])
        np.testing.assert_array_equal(result, expected)

    def test_delay_greater_than_1(self):
        """Non-unit delay: dim=2, delay=2."""
        series = [1, 2, 3, 4, 5, 6]
        result = delay_embedding(series, dim=2, delay=2)
        expected = np.array([
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
        ])
        np.testing.assert_array_equal(result, expected)

    def test_dim_1_returns_column_matrix(self):
        """Edge case: dim=1 should return a column vector."""
        series = [1, 2, 3, 4]
        result = delay_embedding(series, dim=1, delay=1)
        expected = np.array([[1], [2], [3], [4]])
        np.testing.assert_array_equal(result, expected)

    def test_numpy_array_input(self):
        """Function should accept a numpy array directly."""
        series = np.arange(10)
        result = delay_embedding(series, dim=2, delay=3)
        self.assertEqual(result.shape, (7, 2))
        np.testing.assert_array_equal(result[:, 0], series[:7])
        np.testing.assert_array_equal(result[:, 1], series[3:10])

    def test_float_series(self):
        """Function should work with floating-point data."""
        series = [0.5, 1.5, 2.5, 3.5]
        result = delay_embedding(series, dim=2, delay=1)
        expected = np.array([
            [0.5, 1.5],
            [1.5, 2.5],
            [2.5, 3.5],
        ])
        np.testing.assert_array_almost_equal(result, expected)

    def test_output_dtype_matches_input(self):
        """Output dtype should match input dtype."""
        series = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        result = delay_embedding(series, dim=2, delay=1)
        self.assertEqual(result.dtype, np.int32)

    # ------------------------------------------------------------------
    # Error-condition tests
    # ------------------------------------------------------------------

    def test_raises_on_2d_input(self):
        """A 2D array should raise ValueError."""
        series = np.array([[1, 2], [3, 4]])
        with self.assertRaises(ValueError) as cm:
            delay_embedding(series, dim=2, delay=1)
        self.assertIn("1-dimensional", str(cm.exception))

    def test_raises_on_insufficient_length(self):
        """Series too short for given dim/delay should raise ValueError."""
        series = [1, 2, 3]
        with self.assertRaises(ValueError) as cm:
            delay_embedding(series, dim=5, delay=1)
        self.assertIn("too short", str(cm.exception))

    def test_raises_on_exact_boundary_length(self):
        """n_points == 0 should still raise (no valid output rows)."""
        series = [1, 2]
        # len=2, dim=3, delay=1 -> n_points = 2 - 2 * 1 = 0
        with self.assertRaises(ValueError):
            delay_embedding(series, dim=3, delay=1)

    def test_raises_on_empty_series(self):
        """Empty series should raise ValueError."""
        with self.assertRaises(ValueError):
            delay_embedding([], dim=1, delay=1)

    # ------------------------------------------------------------------
    # Shape and consistency checks
    # ------------------------------------------------------------------

    def test_output_shape_formula(self):
        """Verify the documented shape formula holds for random inputs."""
        series = np.random.randn(100)
        for dim in [1, 2, 4, 7]:
            for delay in [1, 2, 5]:
                result = delay_embedding(series, dim=dim, delay=delay)
                expected_rows = 100 - (dim - 1) * delay
                self.assertEqual(result.shape, (expected_rows, dim))

    def test_row_independence(self):
        """Each row should be a valid delayed snapshot of the series."""
        series = np.arange(50)
        dim, delay = 4, 2
        result = delay_embedding(series, dim=dim, delay=delay)
        for i in range(result.shape[0]):
            expected_row = series[i  : i  + dim * delay : delay]
            np.testing.assert_array_equal(result[i], expected_row)


if __name__ == "__main__":
    unittest.main()