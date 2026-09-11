import unittest

import numpy as np
from scipy.spatial.distance import cdist

from nolitisea.core.embed import delay_embedding
from nolitisea.stationarity.recurrence import (
    recurrence_matrix,
    recurrence_matrix_fixed_rr,
)


def _off_diagonal_rr(mask):
    """Fraction of True entries excluding the main diagonal."""
    n = mask.shape[0]
    m = np.asarray(mask, dtype=bool).copy()
    np.fill_diagonal(m, False)
    return int(m.sum()) / (n * (n - 1))


def _upper_triangle_count(mask):
    """Number of True entries in the strict upper triangle."""
    n = mask.shape[0]
    return int(mask[np.triu_indices(n, k=1)].sum())


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


class TestRecurrenceMatrixFixedRR(unittest.TestCase):
    """Unit tests for recurrence_matrix_fixed_rr function."""

    # ------------------------------------------------------------------
    # Happy-path tests: threshold selection
    # ------------------------------------------------------------------

    def test_continuous_data_hits_nearest_attainable_rr(self):
        """On continuous data the realised RR is the nearest attainable value."""
        rng = np.random.default_rng(123)
        series = rng.standard_normal(200)
        n_points = len(series) - 2  # dim=3, delay=1
        granularity = 2.0 / (n_points * (n_points - 1))
        for target in (0.01, 0.05, 0.10, 0.25, 0.50):
            with self.subTest(target=target):
                mask, eps = recurrence_matrix_fixed_rr(
                    series, dim=3, delay=1, target_rr=target
                )
                # Allow one extra granularity step for rare floating-point ties.
                self.assertLessEqual(abs(_off_diagonal_rr(mask) - target), 2 * granularity)
                self.assertGreater(eps, 0.0)
                self.assertTrue(np.isfinite(eps))

    def test_retained_pair_count_is_order_statistic(self):
        """With unique distances exactly round(target * n_pairs) pairs survive."""
        rng = np.random.default_rng(42)
        series = rng.standard_normal(60)
        embedded = delay_embedding(series, 2, 1)
        distances = cdist(embedded, embedded, metric="euclidean")
        upper = distances[np.triu_indices(embedded.shape[0], k=1)]
        # Guard: the test's exact-count claim needs unique distances.
        self.assertEqual(np.unique(upper).size, upper.size)

        n_pairs = upper.size
        for target in (0.03, 0.17, 0.4):
            with self.subTest(target=target):
                mask, eps = recurrence_matrix_fixed_rr(
                    series, dim=2, delay=1, target_rr=target
                )
                expected_count = round(target * n_pairs)
                self.assertEqual(_upper_triangle_count(mask), expected_count)
                # eps is the (k+1)-th smallest distance.
                self.assertAlmostEqual(eps, np.sort(upper)[expected_count])

    def test_target_zero_keeps_only_diagonal(self):
        """target_rr=0 must retain no off-diagonal pair but keep the LOI."""
        rng = np.random.default_rng(7)
        series = rng.standard_normal(80)
        mask, _ = recurrence_matrix_fixed_rr(series, dim=2, delay=1, target_rr=0.0)
        self.assertEqual(_off_diagonal_rr(mask), 0.0)
        self.assertTrue(np.all(np.diag(mask)))
        # eps equals the smallest positive observed distance.
        np.testing.assert_array_equal(mask, np.eye(mask.shape[0], dtype=bool))

    def test_target_one_keeps_every_pair(self):
        """target_rr=1 must retain every pair (eps = inf)."""
        series = [0, 1, 2, 3, 4, 5, 6, 7]
        mask, eps = recurrence_matrix_fixed_rr(series, dim=2, delay=1, target_rr=1.0)
        self.assertTrue(np.all(mask))
        self.assertEqual(_off_diagonal_rr(mask), 1.0)
        self.assertEqual(eps, float("inf"))

    def test_eps_is_an_observed_distance(self):
        """For an interior target the threshold is an observed pair distance."""
        rng = np.random.default_rng(99)
        series = rng.standard_normal(100)
        embedded = delay_embedding(series, 3, 2)
        distances = cdist(embedded, embedded, metric="euclidean")
        upper = distances[np.triu_indices(embedded.shape[0], k=1)]
        _, eps = recurrence_matrix_fixed_rr(series, dim=3, delay=2, target_rr=0.2)
        self.assertIn(eps, set(upper.tolist()))

    def test_mask_is_thresholded_distance_matrix(self):
        """The mask must equal cdist(...) < eps and be symmetric boolean."""
        rng = np.random.default_rng(5)
        series = rng.standard_normal(50)
        mask, eps = recurrence_matrix_fixed_rr(series, dim=2, delay=2, target_rr=0.15)
        embedded = delay_embedding(series, 2, 2)
        expected = cdist(embedded, embedded, metric="euclidean") < eps
        self.assertEqual(mask.dtype, bool)
        np.testing.assert_array_equal(mask, expected)
        np.testing.assert_array_equal(mask, mask.T)

    def test_monotonic_in_target(self):
        """Larger target_rr must never reduce the number of recurrences."""
        rng = np.random.default_rng(11)
        series = rng.standard_normal(120)
        counts = []
        for target in (0.0, 0.05, 0.1, 0.25, 0.5, 0.9, 1.0):
            mask, _ = recurrence_matrix_fixed_rr(series, dim=3, delay=1, target_rr=target)
            counts.append(int(mask.sum()))
        self.assertTrue(np.all(np.diff(counts) >= 0))

    def test_deterministic_output(self):
        """Repeated calls with identical inputs return identical results."""
        rng = np.random.default_rng(3)
        series = rng.standard_normal(70)
        mask1, eps1 = recurrence_matrix_fixed_rr(series, dim=2, delay=1, target_rr=0.12)
        mask2, eps2 = recurrence_matrix_fixed_rr(series, dim=2, delay=1, target_rr=0.12)
        np.testing.assert_array_equal(mask1, mask2)
        self.assertEqual(eps1, eps2)

    def test_metric_argument_is_used(self):
        """The metric must be forwarded to the distance computation."""
        rng = np.random.default_rng(21)
        series = rng.standard_normal(40)
        mask, eps = recurrence_matrix_fixed_rr(
            series, dim=2, delay=1, target_rr=0.2, metric="cityblock"
        )
        embedded = delay_embedding(series, 2, 1)
        expected = cdist(embedded, embedded, metric="cityblock") < eps
        np.testing.assert_array_equal(mask, expected)

    # ------------------------------------------------------------------
    # Tied / quantised distances
    # ------------------------------------------------------------------

    def test_tied_distances_share_boundary_fate(self):
        """Pairs tied at the boundary are kept/excluded together.

        dim=1 embedding of [0, 0, 1, 1] gives upper-triangle distances
        [0, 1, 1, 1, 1, 0] (sorted: [0, 0, 1, 1, 1, 1]).  target=0.5 rounds
        to k=3 pairs, so eps = sorted[3] = 1.0 and the strict threshold keeps
        only the two zero-distance duplicate pairs.
        """
        series = np.array([0.0, 0.0, 1.0, 1.0])
        mask, eps = recurrence_matrix_fixed_rr(series, dim=1, delay=1, target_rr=0.5)
        self.assertEqual(eps, 1.0)
        self.assertEqual(_upper_triangle_count(mask), 2)
        self.assertAlmostEqual(_off_diagonal_rr(mask), 2.0 / 6.0)

    # ------------------------------------------------------------------
    # Edge cases and error handling
    # ------------------------------------------------------------------

    def test_single_embedded_point(self):
        """One embedded point -> 1x1 True matrix and eps 0.0."""
        mask, eps = recurrence_matrix_fixed_rr([1, 2], dim=2, delay=1, target_rr=0.5)
        self.assertEqual(mask.shape, (1, 1))
        self.assertTrue(mask[0, 0])
        self.assertEqual(eps, 0.0)

    def test_constant_series_raises_runtime_error(self):
        """A constant series collapses the state space -> RuntimeError."""
        with self.assertRaises(RuntimeError):
            recurrence_matrix_fixed_rr(np.ones(100), dim=3, delay=2, target_rr=0.05)
        with self.assertRaises(RuntimeError):
            recurrence_matrix_fixed_rr([7] * 20, dim=2, delay=1, target_rr=0.5)

    def test_invalid_target_rr_raises_value_error(self):
        """target_rr outside [0, 1] (including nan) must raise ValueError."""
        series = np.random.default_rng(0).standard_normal(50)
        for bad in (-0.1, 1.01, 2.0, np.nan):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                recurrence_matrix_fixed_rr(series, dim=2, delay=1, target_rr=bad)


if __name__ == "__main__":
    unittest.main()