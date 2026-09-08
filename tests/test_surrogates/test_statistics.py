"""Tests for the statistics in ``nolitisea.surrogates.statistics``.

The brute-force helper transcribes the ``fcerror`` routine with
an explicit O(N^2) Chebyshev neighbour search so the cKDTree-based
production code can be validated on continuous data (exact agreement)
as well as on quantized data containing duplicate delay vectors.
"""

import unittest

import numpy as np

from nolitisea.generate.henon import henon
from nolitisea.surrogates.statistics import predict_stat, time_reversibility

# ---------------------------------------------------------------------------
# Brute-force helpers
# ---------------------------------------------------------------------------


def _brute_fcerror(x, dim, delay, n_forecast, eps):
    """O(N^2) transcription of the ``fcerror`` routine.

    Builds the delay vectors explicitly (oldest-first order),
    finds all Chebyshev ``eps``-neighbours by exhaustive comparison,
    drops the reference point itself, and averages the neighbours'
    ``n_forecast``-steps-ahead values in ascending index order (the
    documented deterministic rule).  Falls back to the series mean when
    the reference point has no other neighbour inside the ball.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    valid_start = (dim - 1) * delay
    n_refs = n - n_forecast - valid_start

    vecs = np.empty((n_refs, dim))
    for i in range(n_refs):
        t = valid_start + i
        for k in range(dim):
            vecs[i, k] = x[t - k * delay]

    mean_x = x.mean()
    sq = np.empty(n_refs)
    for i in range(n_refs):
        dists = np.max(np.abs(vecs - vecs[i]), axis=1)
        nbrs = np.sort(np.nonzero(dists <= eps)[0].astype(np.intp))
        if nbrs.size > 1:
            contrib = nbrs[nbrs != i]
            # rows -> endpoint times t = row + valid_start; the
            # forecast reads y(endpoint + ifc)
            pred = x[contrib + valid_start + n_forecast].sum() / (nbrs.size - 1)
        else:
            pred = mean_x
        sq[i] = (x[i + valid_start + n_forecast] - pred) ** 2
    return float(np.sqrt(sq.mean()))


# ---------------------------------------------------------------------------
# time_reversibility
# ---------------------------------------------------------------------------


class TestTimeReversibility(unittest.TestCase):
    def test_hand_computed_delay_1(self):
        x = np.array([1.0, 2.0, 4.0, 3.0, 5.0])
        # dx = [1, 2, -1, 2]: sum dx^3 = 16, sum dx^2 = 10
        self.assertAlmostEqual(time_reversibility(x), 16.0 / 10.0, places=14)

    def test_hand_computed_delay_2(self):
        x = np.array([1.0, 2.0, 4.0, 3.0, 5.0])
        # dx = x[2:] - x[:-2] = [3, 1, 1]: sum dx^3 = 29, sum dx^2 = 11
        self.assertAlmostEqual(time_reversibility(x, delay=2), 29.0 / 11.0, places=14)

    def test_reversal_antisymmetry(self):
        rng = np.random.default_rng(42)
        x = np.cumsum(rng.normal(size=1000))
        self.assertAlmostEqual(
            time_reversibility(x[::-1]), -time_reversibility(x), places=10
        )

    def test_gaussian_noise_near_zero(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=50000)
        # Time-reversible process: T fluctuates around zero (sigma ~ 0.025).
        self.assertLess(abs(time_reversibility(x)), 0.1)

    def test_henon_clearly_asymmetric(self):
        x, _ = henon(n=20000)
        self.assertGreater(abs(time_reversibility(x)), 0.05)

    def test_constant_series_nan(self):
        self.assertTrue(np.isnan(time_reversibility(np.zeros(50))))

    def test_two_dimensional_input_raises(self):
        with self.assertRaises(ValueError):
            time_reversibility(np.zeros((10, 2)))

    def test_invalid_delay_raises(self):
        with self.assertRaises(ValueError):
            time_reversibility(np.ones(10), delay=0)

    def test_too_short_series_raises(self):
        with self.assertRaises(ValueError):
            time_reversibility(np.ones(3), delay=3)


# ---------------------------------------------------------------------------
# predict_stat
# ---------------------------------------------------------------------------


class TestPredictStatBruteForceAgreement(unittest.TestCase):
    def test_henon_continuous_data_exact(self):
        x, _ = henon(n=300)
        combos = [
            (2, 1, 1, 0.1),
            (3, 2, 1, 0.2),
            (2, 1, 3, 0.15),
            (4, 1, 1, 0.3),
            (2, 1, 1, 5.0),  # whole-box radius: every reference has neighbours
        ]
        for dim, delay, nf, eps in combos:
            with self.subTest(dim=dim, delay=delay, n_forecast=nf, eps=eps):
                got = predict_stat(x, dim, delay, n_forecast=nf, eps=eps)
                ref = _brute_fcerror(x, dim, delay, nf, eps)
                self.assertEqual(got, ref)

    def test_quantized_data_with_duplicates_exact(self):
        # Quantization creates duplicate delay vectors; duplicates inside
        # the eps-ball must be kept (only the reference point itself is
        # excluded).
        rng = np.random.default_rng(7)
        x = np.round(rng.normal(size=120) * 2.0) / 2.0
        for dim, delay, nf, eps in [(2, 1, 1, 0.75), (3, 1, 1, 1.0), (2, 2, 2, 0.5)]:
            with self.subTest(dim=dim, delay=delay, n_forecast=nf, eps=eps):
                got = predict_stat(x, dim, delay, n_forecast=nf, eps=eps)
                ref = _brute_fcerror(x, dim, delay, nf, eps)
                self.assertEqual(got, ref)


class TestPredictStatHandComputed(unittest.TestCase):
    def test_all_neighbours_case(self):
        # eps large enough that every reference sees the other vector.
        x = np.array([1.0, 2.0, 3.0, 4.0])
        # Refs t=1,2; each ball holds both vectors -> single contributing
        # neighbour: t=1 predicts x[3]=4 (target x[2]=3), t=2 predicts
        # x[2]=3 (target x[3]=4) -> RMSE = 1.
        self.assertEqual(predict_stat(x, dim=2, delay=1, eps=10.0), 1.0)

    def test_mean_fallback_isolated_reference(self):
        # Single reference point with no neighbour except itself.
        x = np.array([0.0, 10.0, 1.0])
        # Ref t=1, vector (0, 10): no other vector -> pred = mean(x) = 11/3,
        # target x[2] = 1 -> RMSE = |1 - 11/3| = 8/3.
        self.assertAlmostEqual(
            predict_stat(x, dim=2, delay=1, eps=0.5), 8.0 / 3.0, places=14
        )

    def test_isolated_reference_inside_larger_series(self):
        # t=3 vector (1, 2) is isolated for eps=0.5 while other references
        # also fall back; checked against the brute reference instead of a
        # hand value.
        x = np.array([0.0, 100.0, 1.0, 2.0, 100.0, 3.0])
        got = predict_stat(x, dim=2, delay=1, eps=0.5)
        ref = _brute_fcerror(x, 2, 1, 1, 0.5)
        self.assertEqual(got, ref)


class TestPredictStatRadiusOptions(unittest.TestCase):
    def test_frac_matches_explicit_eps(self):
        x, _ = henon(n=500)
        eps = 0.3 * x.std()  # population std (ddof=0)
        self.assertEqual(
            predict_stat(x, 2, 1, eps=eps), predict_stat(x, 2, 1, frac=0.3)
        )

    def test_frac_overrides_eps(self):
        x, _ = henon(n=500)
        self.assertEqual(
            predict_stat(x, 2, 1, eps=1e-9, frac=0.3),
            predict_stat(x, 2, 1, frac=0.3),
        )

    def test_missing_radius_raises(self):
        x, _ = henon(n=100)
        with self.assertRaises(ValueError):
            predict_stat(x, 2, 1)
        with self.assertRaises(ValueError):
            predict_stat(x, 2, 1, eps=0.0)
        with self.assertRaises(ValueError):
            predict_stat(x, 2, 1, eps=-0.1)
        with self.assertRaises(ValueError):
            predict_stat(x, 2, 1, frac=0.0)


class TestPredictStatValidation(unittest.TestCase):
    def setUp(self):
        self.x, _ = henon(n=100)

    def test_two_dimensional_input_raises(self):
        with self.assertRaises(ValueError):
            predict_stat(np.zeros((10, 2)), 2, 1, eps=0.1)

    def test_invalid_parameters_raise(self):
        for kwargs in [
            {"dim": 0, "delay": 1},
            {"dim": 2, "delay": 0},
            {"dim": 2, "delay": 1, "n_forecast": 0},
        ]:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                predict_stat(self.x, eps=0.1, **kwargs)

    def test_too_short_series_raises(self):
        with self.assertRaises(ValueError):
            predict_stat(np.ones(5), dim=4, delay=2, n_forecast=3, eps=0.1)

    def test_constant_series_runs(self):
        # Every vector is identical -> neighbours everywhere -> no crash.
        stat = predict_stat(np.zeros(50), dim=2, delay=1, eps=0.1)
        self.assertEqual(stat, 0.0)


class TestPredictStatDiscrimination(unittest.TestCase):
    def test_henon_beats_shuffled_surrogates(self):
        x, _ = henon(n=2000)
        stat_orig = predict_stat(x, 2, 1, frac=0.3)
        rng = np.random.default_rng(123)
        shuf = [predict_stat(rng.permutation(x), 2, 1, frac=0.3) for _ in range(5)]
        self.assertLess(stat_orig, sum(shuf) / len(shuf))


if __name__ == "__main__":
    unittest.main()
