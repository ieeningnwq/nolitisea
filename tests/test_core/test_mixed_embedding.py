import unittest

import numpy as np

from nolitisea.core.embed import delay_embedding, mixed_embedding


class TestMixedEmbedding(unittest.TestCase):
    """Unit tests for mixed_embedding function."""

    # ------------------------------------------------------------------
    # Happy-path tests
    # ------------------------------------------------------------------

    def test_single_variable_matches_delay_embedding(self):
        """A single variable should reduce to plain delay_embedding."""
        series = np.arange(10.0)
        result = mixed_embedding([series], [3], [2])
        expected = delay_embedding(series, dim=3, delay=2)
        np.testing.assert_array_equal(result, expected)

    def test_two_variables_equal_dims_and_delays(self):
        """Equal dims/delays: column blocks match per-variable embeddings."""
        x = np.arange(10.0)
        y = np.arange(100.0, 110.0)
        result = mixed_embedding([x, y], [2, 2], [1, 1])
        expected = np.hstack([
            delay_embedding(x, dim=2, delay=1),
            delay_embedding(y, dim=2, delay=1),
        ])
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.shape, (9, 4))

    def test_different_dims_and_delays_per_variable(self):
        """Each variable uses its own dimension and delay."""
        x = np.arange(10.0)
        y = np.arange(100.0, 110.0)
        result = mixed_embedding([x, y], [3, 2], [2, 1])
        self.assertEqual(result.shape, (6, 5))
        np.testing.assert_array_equal(result[:, 0], x[0:6])
        np.testing.assert_array_equal(result[:, 1], x[2:8])
        np.testing.assert_array_equal(result[:, 2], x[4:10])
        np.testing.assert_array_equal(result[:, 3], y[0:6])
        np.testing.assert_array_equal(result[:, 4], y[1:7])

    def test_dim_1_variable_contributes_single_column(self):
        """A dim=1 variable contributes exactly one column."""
        x = np.arange(8.0)
        y = np.arange(50.0, 58.0)
        result = mixed_embedding([x, y], [1, 2], [1, 2])
        # n_points = min(8, 8 - (2 - 1) * 2) = 6
        self.assertEqual(result.shape, (6, 3))
        np.testing.assert_array_equal(result[:, 0], x[0:6])
        np.testing.assert_array_equal(result[:, 1], y[0:6])
        np.testing.assert_array_equal(result[:, 2], y[2:8])

    def test_different_lengths_align_on_min(self):
        """Rows share the base index; n_points is the min across variables."""
        x = np.arange(10.0)
        z = np.arange(100.0, 108.0)  # shorter series
        result = mixed_embedding([x, z], [2, 2], [1, 1])
        self.assertEqual(result.shape, (7, 4))
        np.testing.assert_array_equal(result[:, 0], x[0:7])
        np.testing.assert_array_equal(result[:, 2], z[0:7])

    def test_row_snapshot_semantics(self):
        """Each row holds [x_i(r), x_i(r + tau_i), ...] for every variable."""
        x = np.arange(20.0)
        y = np.arange(200.0, 220.0)
        dims, delays = [3, 2], [2, 3]
        result = mixed_embedding([x, y], dims, delays)
        n_points = result.shape[0]
        for r in range(n_points):
            np.testing.assert_array_equal(
                result[r, : dims[0]], x[r : r + dims[0] * delays[0] : delays[0]]
            )
            np.testing.assert_array_equal(
                result[r, dims[0] :], y[r : r + dims[1] * delays[1] : delays[1]]
            )

    def test_accepts_lists_and_tuples(self):
        """Non-array inputs and tuple sequences should be accepted."""
        result = mixed_embedding(([1, 2, 3, 4], (5, 6, 7, 8)), (2, 1), (1, 1))
        expected = np.array([[1.0, 2.0, 5.0], [2.0, 3.0, 6.0], [3.0, 4.0, 7.0]])
        np.testing.assert_array_equal(result, expected)

    # ------------------------------------------------------------------
    # Error-condition tests
    # ------------------------------------------------------------------

    def test_raises_on_length_mismatch(self):
        """series_list, dims and delays must have the same length."""
        x = np.arange(10.0)
        with self.assertRaises(ValueError) as cm:
            mixed_embedding([x], [2, 2], [1])
        self.assertIn("same length", str(cm.exception))

    def test_raises_on_empty_input(self):
        """At least one series is required."""
        with self.assertRaises(ValueError) as cm:
            mixed_embedding([], [], [])
        self.assertIn("At least one series", str(cm.exception))

    def test_raises_on_dim_below_one(self):
        """Embedding dimension must be >= 1."""
        x = np.arange(10.0)
        with self.assertRaises(ValueError) as cm:
            mixed_embedding([x], [0], [1])
        self.assertIn("dimension", str(cm.exception))

    def test_raises_on_delay_below_one(self):
        """Delay must be >= 1."""
        x = np.arange(10.0)
        with self.assertRaises(ValueError) as cm:
            mixed_embedding([x], [2], [0])
        self.assertIn("delay", str(cm.exception))

    def test_raises_on_non_1d_series(self):
        """A 2D series should raise ValueError with the variable index."""
        x = np.array([[1, 2], [3, 4]])
        with self.assertRaises(ValueError) as cm:
            mixed_embedding([x], [2], [1])
        self.assertIn("variable 0", str(cm.exception))
        self.assertIn("1-dimensional", str(cm.exception))

    def test_raises_on_insufficient_length(self):
        """A series too short for its embedding should raise ValueError."""
        x = np.arange(3.0)
        with self.assertRaises(ValueError) as cm:
            mixed_embedding([x], [5], [1])
        self.assertIn("too short", str(cm.exception))

    def test_error_message_contains_variable_index(self):
        """The failing variable index appears in the message."""
        good = np.arange(10.0)
        bad = np.arange(2.0)
        with self.assertRaises(ValueError) as cm:
            mixed_embedding([good, bad], [2, 5], [1, 1])
        self.assertIn("variable 1", str(cm.exception))

    # ------------------------------------------------------------------
    # Shape and consistency checks
    # ------------------------------------------------------------------

    def test_output_shape_formula(self):
        """Verify the documented shape formula for random inputs."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(100)
        y = rng.standard_normal(90)
        z = rng.standard_normal(80)
        for dims, delays in [([1, 1, 1], [1, 1, 1]), ([3, 2, 4], [1, 3, 2]),
                             ([2, 5, 2], [4, 1, 3])]:
            result = mixed_embedding([x, y, z], dims, delays)
            n_points = min(
                len(s) - (m - 1) * tau
                for s, m, tau in zip([x, y, z], dims, delays)
            )
            self.assertEqual(result.shape, (n_points, sum(dims)))

    def test_consistency_with_delay_embedding_blocks(self):
        """Column blocks must equal truncated per-variable embeddings."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal(60)
        y = rng.standard_normal(55)
        dims, delays = [4, 3], [2, 1]
        result = mixed_embedding([x, y], dims, delays)
        n_points = result.shape[0]
        block_x = delay_embedding(x, dim=dims[0], delay=delays[0])[:n_points]
        block_y = delay_embedding(y, dim=dims[1], delay=delays[1])[:n_points]
        np.testing.assert_array_equal(result[:, : dims[0]], block_x)
        np.testing.assert_array_equal(result[:, dims[0] :], block_y)


if __name__ == "__main__":
    unittest.main()
