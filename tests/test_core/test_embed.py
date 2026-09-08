"""Tests for :mod:`nolitisea.core.embed`.

Covers :func:`delay_embedding`, :func:`mixed_embedding` and
:func:`delay_vectors`.
"""

import unittest

import numpy as np

from nolitisea.core.embed import delay_embedding, delay_vectors, mixed_embedding

# ---------------------------------------------------------------------------
# delay_embedding
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# mixed_embedding
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# delay_vectors
# ---------------------------------------------------------------------------


class TestDelayVectors(unittest.TestCase):
    """Unit tests for delay_vectors."""

    # ------------------------------------------------------------------
    # Happy-path tests
    # ------------------------------------------------------------------

    def test_univariate_default_embdim_2(self):
        """Defaults: embdim=2, delay=1; vectors anchored at the newest sample."""
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
