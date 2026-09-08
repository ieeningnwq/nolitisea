"""Tests for nolitisea.entropy.transfer_entropy.

The brute-force reference transcribes the histogram TE estimator with
an explicit per-sample dict accumulation, so the vectorized production
code (np.unique packing) can be validated for exact agreement on small
crafted series.
"""

import unittest

import numpy as np

from nolitisea.entropy.transfer_entropy import (
    _digitize,
    _sample_counts,
    transfer_entropy,
)

# ---------------------------------------------------------------------------
# Brute-force reference
# ---------------------------------------------------------------------------


def _brute_transfer_entropy(source, target, k=1, l=1, h=1, bins=10):
    """O(n^2) dict-based histogram TE estimator.

    Builds delay vectors explicitly, digitizes each scalar variable
    with the same equi-width rule as _digitize, and accumulates the four
    required joint distributions in Python dicts.  Computes
    TE = (1/n) * sum_s log2[ n_full * n_yh / (n_yh_xh * n_yf_yh) ]
    per sample, which is numerically equivalent to the standard
    p * log2(ratio) histogram formula.
    """
    x = np.asarray(source, dtype=np.float64).ravel()
    y = np.asarray(target, dtype=np.float64).ravel()
    N = y.size
    max_hist = max(k - 1, l - 1)
    n_samples = N - max_hist - h

    # Build histories and the future target value.
    y_hist = np.column_stack(
        [y[max_hist - i : max_hist - i + n_samples] for i in range(k)]
    )
    x_hist = np.column_stack(
        [x[max_hist - i : max_hist - i + n_samples] for i in range(l)]
    )
    y_future = y[max_hist + h : max_hist + h + n_samples]

    def dig(arr):
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.int64)
        edges = np.linspace(lo, hi, bins + 1)[1:-1]
        return np.clip(np.digitize(arr, edges), 0, bins - 1).astype(np.int64)

    y_h_d = [dig(y_hist[:, i]) for i in range(k)]
    x_h_d = [dig(x_hist[:, i]) for i in range(l)]
    y_f_d = dig(y_future)

    def key_of(indices):
        return tuple(int(i) for i in indices)

    full_counts, yf_yh_counts, yh_xh_counts, yh_counts = {}, {}, {}, {}
    for s in range(n_samples):
        kf = key_of([y_f_d[s]] + [y_h_d[j][s] for j in range(k)]
                    + [x_h_d[j][s] for j in range(l)])
        kyf_yh = key_of([y_f_d[s]] + [y_h_d[j][s] for j in range(k)])
        kyh_xh = key_of([y_h_d[j][s] for j in range(k)]
                        + [x_h_d[j][s] for j in range(l)])
        kyh = key_of([y_h_d[j][s] for j in range(k)])

        full_counts[kf] = full_counts.get(kf, 0) + 1
        yf_yh_counts[kyf_yh] = yf_yh_counts.get(kyf_yh, 0) + 1
        yh_xh_counts[kyh_xh] = yh_xh_counts.get(kyh_xh, 0) + 1
        yh_counts[kyh] = yh_counts.get(kyh, 0) + 1

    te = 0.0
    for s in range(n_samples):
        kf = key_of([y_f_d[s]] + [y_h_d[j][s] for j in range(k)]
                    + [x_h_d[j][s] for j in range(l)])
        kyf_yh = key_of([y_f_d[s]] + [y_h_d[j][s] for j in range(k)])
        kyh_xh = key_of([y_h_d[j][s] for j in range(k)]
                        + [x_h_d[j][s] for j in range(l)])
        kyh = key_of([y_h_d[j][s] for j in range(k)])
        ratio = (full_counts[kf] * yh_counts[kyh]
                 / (yh_xh_counts[kyh_xh] * yf_yh_counts[kyf_yh]))
        te += np.log2(ratio)
    return te / n_samples


# ---------------------------------------------------------------------------
# Helpers _digitize and _sample_counts
# ---------------------------------------------------------------------------


class TestDigitize(unittest.TestCase):
    """Tests for the _digitize helper."""

    def test_constant_maps_to_zero(self):
        x = np.full(20, 3.5)
        idx = _digitize(x, 5)
        np.testing.assert_array_equal(idx, np.zeros(20, dtype=np.int64))

    def test_uniform_range(self):
        # [0, 10) with 5 bins -> idx = floor(x/2), clipped to [0, 4].
        x = np.array([0.0, 1.9, 2.0, 3.9, 8.0, 10.0])
        idx = _digitize(x, 5)
        self.assertEqual(idx.tolist(), [0, 0, 1, 1, 4, 4])

    def test_output_dtype(self):
        idx = _digitize(np.arange(10.0), 4)
        self.assertEqual(idx.dtype, np.int64)


class TestSampleCounts(unittest.TestCase):
    """Tests for the _sample_counts helper."""

    def test_all_unique(self):
        # Every sample has a distinct combination -> all counts = 1.
        a = np.arange(10, dtype=np.int64)
        b = np.arange(10, dtype=np.int64) * 100
        c = _sample_counts(a, b)
        np.testing.assert_array_equal(c, np.ones(10))

    def test_all_identical(self):
        # Every sample in the same cell -> all counts = n.
        a = np.zeros(10, dtype=np.int64)
        c = _sample_counts(a)
        np.testing.assert_array_equal(c, np.full(10, 10.0))

    def test_partial_grouping(self):
        # 4 samples: two pairs with identical keys -> counts [2, 2, 2, 2].
        a = np.array([0, 0, 1, 1], dtype=np.int64)
        b = np.array([10, 10, 20, 20], dtype=np.int64)
        c = _sample_counts(a, b)
        np.testing.assert_array_equal(c, np.full(4, 2.0))

    def test_empty(self):
        c = _sample_counts(np.array([], dtype=np.int64))
        self.assertEqual(c.size, 0)


# ---------------------------------------------------------------------------
# Brute-force agreement
# ---------------------------------------------------------------------------


class TestTransferEntropyBruteForce(unittest.TestCase):
    """Compare vectorized TE against the dict-based brute-force reference."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.x_coupled = rng.standard_normal(80)
        cls.y_coupled = cls.x_coupled * 0.7 + 0.3 * rng.standard_normal(80)
        cls.x_long = rng.standard_normal(200)
        cls.y_long = rng.standard_normal(200)

    def test_default_params(self):
        got = transfer_entropy(self.x_coupled, self.y_coupled)
        ref = _brute_transfer_entropy(self.x_coupled, self.y_coupled)
        self.assertAlmostEqual(got, ref, places=12)

    def test_bins_variation(self):
        for bins in [2, 3, 5, 8]:
            with self.subTest(bins=bins):
                got = transfer_entropy(self.x_coupled, self.y_coupled,
                                       bins=bins)
                ref = _brute_transfer_entropy(self.x_coupled, self.y_coupled,
                                              bins=bins)
                self.assertAlmostEqual(got, ref, places=12)

    def test_k_l_h_variation(self):
        for k, l, h in [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2),
                        (2, 2, 1), (2, 1, 2), (3, 2, 1), (2, 3, 2)]:
            with self.subTest(k=k, l=l, h=h):
                got = transfer_entropy(self.x_long, self.y_long,
                                       k=k, l=l, h=h, bins=4)
                ref = _brute_transfer_entropy(self.x_long, self.y_long,
                                             k=k, l=l, h=h, bins=4)
                self.assertAlmostEqual(got, ref, places=12)

    def test_constant_source(self):
        # Constant source collapses to a single bin -> TE = 0 (no
        # information beyond the target's own past).
        x = np.full(50, 2.0)
        y = np.linspace(0, 10, 50)
        got = transfer_entropy(x, y, bins=5)
        ref = _brute_transfer_entropy(x, y, bins=5)
        self.assertAlmostEqual(got, ref, places=12)
        self.assertEqual(got, 0.0)

    def test_constant_target(self):
        # Constant target: y_future and y_hist are constant -> all cells
        # collapse and TE = 0.
        x = np.linspace(0, 10, 50)
        y = np.full(50, 2.0)
        got = transfer_entropy(x, y, bins=5)
        ref = _brute_transfer_entropy(x, y, bins=5)
        self.assertAlmostEqual(got, ref, places=12)
        self.assertEqual(got, 0.0)

    def test_identical_series(self):
        # X == Y: source and target identical, k=l=1, h=1.
        # By symmetry and the chain rule, TE(X->X) measures the
        # information that the present value carries about the future
        # beyond what is already in the past (with k=1, this is
        # typically nonzero for autocorrelated series).  Just check
        # exact agreement with brute force.
        rng = np.random.default_rng(7)
        x = np.cumsum(rng.standard_normal(80))
        got = transfer_entropy(x, x, bins=4)
        ref = _brute_transfer_entropy(x, x, bins=4)
        self.assertAlmostEqual(got, ref, places=12)


# ---------------------------------------------------------------------------
# Hand-computed values
# ---------------------------------------------------------------------------


class TestTransferEntropyHandComputed(unittest.TestCase):
    """Validate against hand-computed values on tiny series."""

    def test_two_bin_symmetric_series(self):
        # Construct a small series where the joint counts are tractable.
        # X: [0, 1, 0, 1, 0, 1] (alternating), Y = X shifted by 1
        # (Y[t] = X[t-1]).  With 2 bins, k=l=h=1:
        # Y future = Y[1:] = [1, 0, 1, 0, 1], Y past = Y[:-1] = [0,1,0,1,0]
        # X past = X[:-1]  = [0, 1, 0, 1, 0]
        # Every (y_f, y_h, x_h) cell: (1,0,0), (0,1,1), (1,0,0), (0,1,1),
        # (1,0,0).  Three cells repeat: (1,0,0) x3, (0,1,1) x2.
        # Brute reference computes the exact value; we just check it
        # matches and is positive.
        x = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])  # y[t] = x[t-1]
        got = transfer_entropy(x, y, k=1, l=1, h=1, bins=2)
        ref = _brute_transfer_entropy(x, y, k=1, l=1, h=1, bins=2)
        self.assertAlmostEqual(got, ref, places=12)
        # y_h=0 -> y_f=1 (3 cases); y_h=1 -> y_f=0 (2 cases).  X_h fully
        # determines y_h (x_h == y_h here), so conditioning on y_h already
        # captures x_h -> TE should be 0 (x_h adds nothing beyond y_h).
        self.assertEqual(got, 0.0)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestTransferEntropyValidation(unittest.TestCase):
    """Tests for parameter validation."""

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            transfer_entropy(np.arange(10.0), np.arange(11.0))

    def test_invalid_k_raises(self):
        with self.assertRaises(ValueError):
            transfer_entropy(np.arange(20.0), np.arange(20.0), k=0)

    def test_invalid_l_raises(self):
        with self.assertRaises(ValueError):
            transfer_entropy(np.arange(20.0), np.arange(20.0), l=0)

    def test_invalid_h_raises(self):
        with self.assertRaises(ValueError):
            transfer_entropy(np.arange(20.0), np.arange(20.0), h=0)

    def test_invalid_bins_raises(self):
        with self.assertRaises(ValueError):
            transfer_entropy(np.arange(20.0), np.arange(20.0), bins=0)

    def test_too_short_series_raises(self):
        # N=3, k=2, l=2, h=2 -> max_hist=1, n_samples = 3 - 1 - 2 = 0.
        with self.assertRaises(ValueError):
            transfer_entropy(np.arange(3.0), np.arange(3.0), k=2, l=2, h=2)

    def test_too_short_for_h_raises(self):
        # N=5, k=1, l=1, h=4 -> n_samples = 5 - 0 - 4 = 1 (ok)
        # N=5, h=5 -> n_samples = 0 (raises)
        with self.assertRaises(ValueError):
            transfer_entropy(np.arange(5.0), np.arange(5.0), h=5)

    def test_2d_input_accepted(self):
        # 2-D inputs are raveled to 1-D; should not raise.
        x = np.arange(20.0).reshape(2, 10)
        y = np.arange(20.0).reshape(2, 10)
        # Both ravel to [0..19], equal length -> valid.
        te = transfer_entropy(x, y, bins=4)
        self.assertIsInstance(te, float)


# ---------------------------------------------------------------------------
# Physical / statistical properties
# ---------------------------------------------------------------------------


class TestTransferEntropyProperties(unittest.TestCase):
    """Tests for non-negativity, determinism, and basic properties."""

    def test_returns_float(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(100)
        y = rng.standard_normal(100)
        self.assertIsInstance(transfer_entropy(x, y, bins=5), float)

    def test_non_negative(self):
        # TE is a conditional MI; estimator can produce tiny negative
        # values from finite-sample bias on independent data, but on
        # coupled data it must be >= 0.
        rng = np.random.default_rng(0)
        x = np.zeros(500)
        y = np.zeros(500)
        for t in range(499):
            x[t + 1] = 0.7 * x[t] + 0.3 * rng.standard_normal()
            y[t + 1] = 0.4 * y[t] + 0.6 * x[t] + 0.2 * rng.standard_normal()
        te = transfer_entropy(x, y, bins=8)
        self.assertGreaterEqual(te, 0.0)

    def test_deterministic_output(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        y = 0.5 * x + 0.5 * rng.standard_normal(200)
        te1 = transfer_entropy(x, y, bins=6)
        te2 = transfer_entropy(x, y, bins=6)
        self.assertEqual(te1, te2)

    def test_independent_series_near_zero(self):
        # Two independent white-noise series: TE should be small.
        rng = np.random.default_rng(0)
        x = rng.standard_normal(5000)
        y = rng.standard_normal(5000)
        te = transfer_entropy(x, y, bins=8)
        self.assertLess(abs(te), 0.1)

    def test_coupled_series_positive(self):
        # Y depends on X: TE(X->Y) should be clearly positive.
        rng = np.random.default_rng(0)
        n = 5000
        x = np.zeros(n)
        y = np.zeros(n)
        for t in range(n - 1):
            x[t + 1] = 0.8 * x[t] + 0.2 * rng.standard_normal()
            y[t + 1] = 0.5 * y[t] + 0.5 * x[t] + 0.2 * rng.standard_normal()
        te = transfer_entropy(x, y, bins=8)
        self.assertGreater(te, 0.05)


# ---------------------------------------------------------------------------
# Directionality
# ---------------------------------------------------------------------------


class TestTransferEntropyDirectionality(unittest.TestCase):
    """Tests for the directed (asymmetric) nature of transfer entropy."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        n = 5000
        x = np.zeros(n)
        y = np.zeros(n)
        for t in range(n - 1):
            x[t + 1] = 0.8 * x[t] + 0.2 * rng.standard_normal()
            y[t + 1] = 0.5 * y[t] + 0.5 * x[t] + 0.2 * rng.standard_normal()
        cls.x = x
        cls.y = y

    def test_forward_direction_dominates(self):
        # X drives Y -> TE(X->Y) >> TE(Y->X).
        te_xy = transfer_entropy(self.x, self.y, bins=8)
        te_yx = transfer_entropy(self.y, self.x, bins=8)
        self.assertGreater(te_xy, te_yx)
        self.assertGreater(te_xy / max(te_yx, 1e-12), 3.0)

    def test_reverse_direction_small(self):
        # The spurious reverse TE from finite-sample bias should be small.
        te_yx = transfer_entropy(self.y, self.x, bins=8)
        self.assertLess(te_yx, 0.1)

    def test_coupled_greater_than_independent(self):
        # TE(X->Y) on coupled data >> TE on independent data of the same
        # length.
        rng = np.random.default_rng(123)
        x_ind = rng.standard_normal(5000)
        y_ind = rng.standard_normal(5000)
        te_coupled = transfer_entropy(self.x, self.y, bins=8)
        te_indep = transfer_entropy(x_ind, y_ind, bins=8)
        self.assertGreater(te_coupled, 5 * te_indep)


# ---------------------------------------------------------------------------
# Parameter sensitivity
# ---------------------------------------------------------------------------


class TestTransferEntropyParameters(unittest.TestCase):
    """Tests for the effect of k, l, h, and bins on the estimate."""

    def test_bins_does_not_crash(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        y = 0.5 * x + 0.5 * rng.standard_normal(200)
        for bins in [2, 5, 10, 20]:
            with self.subTest(bins=bins):
                te = transfer_entropy(x, y, bins=bins)
                self.assertTrue(np.isfinite(te))

    def test_k_l_h_variation_runs(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(200)
        y = 0.5 * x + 0.5 * rng.standard_normal(200)
        for k, l, h in [(1, 1, 1), (2, 2, 2), (3, 1, 1), (1, 3, 1)]:
            with self.subTest(k=k, l=l, h=h):
                te = transfer_entropy(x, y, k=k, l=l, h=h, bins=4)
                self.assertTrue(np.isfinite(te))

    def test_h_increases_memory_decay(self):
        # For a coupled AR(1) with coef 0.5, larger h reduces the
        # information that the source carries about the target's future.
        # TE at h=1 should generally exceed TE at h=4 (longer horizon
        # loses predictability).
        rng = np.random.default_rng(0)
        n = 5000
        x = np.zeros(n)
        y = np.zeros(n)
        for t in range(n - 1):
            x[t + 1] = 0.8 * x[t] + 0.2 * rng.standard_normal()
            y[t + 1] = 0.5 * y[t] + 0.5 * x[t] + 0.2 * rng.standard_normal()
        te_h1 = transfer_entropy(x, y, h=1, bins=8)
        te_h4 = transfer_entropy(x, y, h=4, bins=8)
        self.assertGreater(te_h1, te_h4)


if __name__ == "__main__":
    unittest.main()
