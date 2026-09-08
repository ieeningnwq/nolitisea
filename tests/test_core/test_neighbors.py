"""Tests for :func:`nolitisea.core.neighbors.find_neighbors`."""

import unittest

import numpy as np

from nolitisea.core.neighbors import find_neighbors

# ---------------------------------------------------------------------------
# Brute-force reference
# ---------------------------------------------------------------------------


def _brute_knn(y, metric, theiler, k):
    """O(N^2) transcription of ``find_neighbors``.

    For each point, compute distances to every other point under
    ``metric``, keep the ``k`` smallest, filter out self-matches,
    zero distances and points inside the Theiler window, and return
    the first surviving neighbour.  ``theiler`` means points with
    ``|j - i| <= theiler`` are excluded.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if metric == "cityblock":
        p = 1.0
    elif metric == "euclidean":
        p = 2.0
    elif metric == "chebyshev":
        p = np.inf
    else:
        raise ValueError(metric)

    indices = np.empty(n, dtype=np.intp)
    dists = np.empty(n, dtype=np.float64)

    for i in range(n):
        # All pairwise distances via cKDTree-style (recompute to stay pure).
        diffs = np.abs(y - y[i]) if p == np.inf else np.abs(y - y[i])  # noqa: RUF034
        if p == np.inf:
            d = diffs.max(axis=1)
        elif p == 1.0:
            d = diffs.sum(axis=1)
        elif p == 2.0:
            d = np.sqrt((diffs ** 2).sum(axis=1))
        else:
            raise RuntimeError("unreachable")

        order = np.argsort(d, kind="mergesort")  # stable tie-break
        kept = 0
        found = False
        for j in order:
            if d[j] == 0.0:
                continue
            if abs(j - i) <= theiler:
                continue
            kept += 1
            if kept == 1:
                indices[i] = j
                dists[i] = d[j]
                found = True
                break
        if not found:
            raise RuntimeError("no valid neighbour found (brute)")
    return indices, dists


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestMaxnumValidation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = rng.normal(size=(50, 3))

    def test_none_defaults_to_tiseran_heuristic(self):
        # theiler=0 -> default = 2*(0+1)+1 = 3, passes theil+2=2 bound.
        idx, _ = find_neighbors(self.y, maxnum=None)
        self.assertEqual(idx.shape, (50,))

    def test_none_default_respects_theiler(self):
        # theiler=5 -> default = 2*(5+1)+1 = 13, passes theil+2=7 bound.
        idx, _ = find_neighbors(self.y, theiler=5)
        self.assertEqual(idx.shape, (50,))
        # All returned neighbours must lie outside the Theiler window.
        self.assertTrue(np.all(np.abs(idx - np.arange(len(idx))) > 5))

    def test_maxnum_zero_raises(self):
        with self.assertRaises(ValueError) as ctx:
            find_neighbors(self.y, maxnum=0)
        self.assertIn("theiler + 2", str(ctx.exception))

    def test_maxnum_one_raises(self):
        with self.assertRaises(ValueError) as ctx:
            find_neighbors(self.y, maxnum=1)
        self.assertIn("theiler + 2", str(ctx.exception))

    def test_maxnum_negative_raises(self):
        with self.assertRaises(ValueError) as ctx:
            find_neighbors(self.y, maxnum=-7)
        self.assertIn("theiler + 2", str(ctx.exception))

    def test_maxnum_below_theiler_bound_raises(self):
        # theiler=5, theiler+2=7: 5 and 6 must raise.
        for m in [5, 6]:
            with self.subTest(maxnum=m):
                with self.assertRaises(ValueError) as ctx:
                    find_neighbors(self.y, theiler=5, maxnum=m)
                self.assertIn("theiler + 2", str(ctx.exception))

    def test_maxnum_equal_theiler_plus_two_is_ok(self):
        # Lower bound: self (1) + theiler (5) + 1 valid = 7.
        idx, _ = find_neighbors(self.y, theiler=5, maxnum=7)
        self.assertTrue(np.all(np.abs(idx - np.arange(len(idx))) > 5))

    def test_maxnum_equals_n_raises(self):
        with self.assertRaises(ValueError):
            find_neighbors(self.y, maxnum=len(self.y))

    def test_maxnum_above_n_raises(self):
        with self.assertRaises(ValueError):
            find_neighbors(self.y, maxnum=10000)


class TestMetricValidation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.y = rng.normal(size=(30, 2))

    def test_valid_metrics(self):
        for metric in ["chebyshev", "euclidean", "cityblock"]:
            with self.subTest(metric=metric):
                idx, _ = find_neighbors(self.y, metric=metric, maxnum=5)
                self.assertEqual(idx.shape, (30,))

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            find_neighbors(self.y, metric="cosine", maxnum=5)


class TestBruteForceAgreement(unittest.TestCase):
    def test_chebyshev_default_matches_brute(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=(80, 3))
        for theiler in [0, 3, 7]:
            with self.subTest(theiler=theiler):
                idx_p, d_p = find_neighbors(y, theiler=theiler)
                idx_b, d_b = _brute_knn(y, "chebyshev", theiler, k=13)
                np.testing.assert_array_equal(idx_p, idx_b)
                np.testing.assert_allclose(d_p, d_b)

    def test_euclidean_matches_brute(self):
        rng = np.random.default_rng(2)
        y = rng.normal(size=(60, 2))
        idx_p, d_p = find_neighbors(y, metric="euclidean", theiler=2)
        idx_b, d_b = _brute_knn(y, "euclidean", 2, k=7)
        np.testing.assert_array_equal(idx_p, idx_b)
        np.testing.assert_allclose(d_p, d_b)

    def test_cityblock_matches_brute(self):
        rng = np.random.default_rng(3)
        y = rng.normal(size=(60, 2))
        idx_p, d_p = find_neighbors(y, metric="cityblock", theiler=0)
        idx_b, d_b = _brute_knn(y, "cityblock", 0, k=3)
        np.testing.assert_array_equal(idx_p, idx_b)
        np.testing.assert_allclose(d_p, d_b)

    def test_all_indices_outside_theiler_window(self):
        rng = np.random.default_rng(5)
        y = rng.normal(size=(100, 4))
        for theiler in [0, 5, 10]:
            with self.subTest(theiler=theiler):
                idx, d = find_neighbors(y, theiler=theiler)
                self.assertTrue(np.all(np.abs(idx - np.arange(len(idx))) > theiler))
                self.assertTrue(np.all(d > 0))


class TestRegressionOnRealData(unittest.TestCase):
    def test_lorenz_embedding_defaults_work(self):
        # The default call path used by false_nearest.kennel_method must
        # succeed on a simple 2-D embedding of a clean sine.
        t = np.linspace(0, 20, 400)
        x = np.sin(t)
        emb = np.column_stack([x[:-1], x[1:]])
        idx, _ = find_neighbors(emb)
        self.assertEqual(idx.shape, (399,))
        self.assertTrue(np.all(idx != np.arange(len(idx))))


if __name__ == "__main__":
    unittest.main()
