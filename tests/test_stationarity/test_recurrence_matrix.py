import unittest

import numpy as np
from scipy.spatial.distance import cdist

from nolitisea.core.embed import delay_embedding
from nolitisea.stationarity.recurrence import recurrence_matrix


class TestRecurrenceMatrix(unittest.TestCase):
    """Unit tests for recurrence_matrix function."""

    # ------------------------------------------------------------------
    # Happy-path tests
    # ------------------------------------------------------------------

    def test_basic_recurrence_structure(self):
        """Verify recurrence matrix correctly flags points within eps."""
        series = [0, 1, 2, 3, 4]
        eps = 1.5
        result = recurrence_matrix(series, dim=2, delay=1, eps=eps)

        # Expected: distance between [i, i+1] and [j, j+1] is sqrt((i-j)^2 + (i-j)^2)
        # = sqrt(2) * |i - j|
        # So R[i,j] = True iff sqrt(2)*|i-j| < 1.5  => |i-j| <= 1
        expected = np.array([
        [True,  True,  False, False],
        [True,  True,  True,  False],
        [False, True,  True,  True ],
        [False, False, True,  True ],
    ])
        np.testing.assert_array_equal(result, expected)

    def test_diagonal_is_all_true(self):
        """The main diagonal must always be True (every point recurs with itself)."""
        series = np.random.randn(50)
        result = recurrence_matrix(series, dim=3, delay=2, eps=0.5)
        np.testing.assert_array_equal(np.diag(result), np.ones(46, dtype=bool))

    def test_symmetric_matrix(self):
        """Recurrence matrix must be symmetric (distance is symmetric)."""
        series = [1, 5, 2, 8, 3, 9, 4, 7]
        result = recurrence_matrix(series, dim=2, delay=1, eps=3.0)
        np.testing.assert_array_equal(result, result.T)

    def test_eps_zero_allows_only_self_recurrence(self):
        """With eps=0, only the diagonal should be True."""
        series = [1, 2, 3, 4, 5]
        result = recurrence_matrix(series, dim=2, delay=1, eps=0.000001)
        expected = np.eye(result.shape[0], dtype=bool)
        np.testing.assert_array_equal(result, expected)

    def test_large_eps_all_true(self):
        """With a very large eps, all entries should be True."""
        series = [1, 2, 3]
        result = recurrence_matrix(series, dim=1, delay=1, eps=1000.0)
        self.assertTrue(np.all(result))

    def test_output_shape_matches_embedding_length(self):
        """Output shape must be (n_points, n_points)."""
        series = np.arange(30)
        dim, delay, eps = 4, 3, 1.0
        embedded = delay_embedding(series, dim, delay)
        n_points = embedded.shape[0]
        result = recurrence_matrix(series, dim=dim, delay=delay, eps=eps)
        self.assertEqual(result.shape, (n_points, n_points))

    def test_output_is_boolean(self):
        """Returned matrix must have boolean dtype."""
        series = [1, 2, 3, 4, 5, 6]
        result = recurrence_matrix(series, dim=2, delay=1, eps=1.0)
        self.assertEqual(result.dtype, bool)

    def test_manhattan_metric(self):
        """Verify correct behaviour with Manhattan (cityblock) metric."""
        series = [0, 1, 2, 3]
        # Embedded: [[0,1],[1,2],[2,3]]
        # Manhattan distance between row 0 and row 2: |0-2| + |1-3| = 4
        result = recurrence_matrix(series, dim=2, delay=1, eps=3.5, metric="cityblock")
        # Distance < 3.5 means only adjacent and self pairs
        self.assertTrue(result[0, 0])   # self
        self.assertTrue(result[0, 1])   # adjacent
        self.assertFalse(result[0, 2])  # too far (dist=4)

    def test_single_point_series(self):
        """Series that yields exactly 1 embedded point -> 1x1 True matrix."""
        series = [1, 2]  # dim=2, delay=1 -> 1 point: [[1,2]]
        result = recurrence_matrix(series, dim=2, delay=1, eps=1.0)
        self.assertEqual(result.shape, (1, 1))
        self.assertTrue(result[0, 0])

    # ------------------------------------------------------------------
    # Consistency / cross-check tests
    # ------------------------------------------------------------------

    def test_matches_manual_cdist_computation(self):
        """Result should match a manual cdist + threshold computation."""
        series = np.random.randn(20)
        dim, delay, eps = 3, 2, 0.8
        embedded = delay_embedding(series, dim, delay)
        expected = cdist(embedded, embedded, metric="euclidean") < eps
        result = recurrence_matrix(series, dim=dim, delay=delay, eps=eps, metric="euclidean")
        np.testing.assert_array_equal(result, expected)

    def test_recurrence_count_increases_as_eps_grows(self):
        """Increasing eps should monotonically increase the number of True entries."""
        series = np.random.randn(15)
        prev_count = 0
        for eps in [0.1, 0.5, 1.0, 2.0, 5.0]:
            result = recurrence_matrix(series, dim=2, delay=1, eps=eps)
            count = np.sum(result)
            self.assertGreaterEqual(count, prev_count)
            prev_count = count

    # ------------------------------------------------------------------
    # Edge-case / error-propagation tests
    # ------------------------------------------------------------------

    def test_propagates_delay_embedding_errors(self):
        """Errors from delay_embedding (e.g. bad input) should propagate."""
        # 2D input should raise ValueError from delay_embedding
        series = np.array([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            recurrence_matrix(series, dim=2, delay=1, eps=1.0)

    def test_propagates_insufficient_length_error(self):
        """Too-short series should raise ValueError."""
        series = [1, 2]
        with self.assertRaises(ValueError):
            recurrence_matrix(series, dim=5, delay=1, eps=1.0)

    def test_empty_series_raises(self):
        """Empty input should raise ValueError."""
        with self.assertRaises(ValueError):
            recurrence_matrix([], dim=1, delay=1, eps=1.0)

    def test_negative_eps_raises_or_gants_empty(self):
        """Negative eps should produce an all-False matrix (or raise)."""
        series = [1, 2, 3, 4]
        result = recurrence_matrix(series, dim=2, delay=1, eps=-1.0)
        # Distance is always >= 0, so < -1 is impossible
        self.assertFalse(np.any(result))


if __name__ == "__main__":
    unittest.main()