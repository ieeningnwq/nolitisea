"""Tests for nolitisea.entropy.permutation_entropy."""

import math
import unittest

import numpy as np

from nolitisea.entropy.permutation_entropy import (
    _ordinal_codes,
    permutation_entropy,
)


def _brute_permutation_entropy(series, order, delay, normalize):
    """Independent brute-force permutation entropy for a single series.

    Windows are enumerated explicitly; each window is mapped to the
    tuple of indices that sorts its values, with ties resolved by
    position (earlier index first), matching the documented tie rule.
    Pattern frequencies are counted with a dict.
    """
    x = np.asarray(series, dtype=np.float64)
    n = len(x)
    n_windows = n - (order - 1) * delay
    counts = {}
    for t in range(n_windows):
        window = [x[t + j * delay] for j in range(order)]
        pattern = tuple(sorted(range(order), key=lambda j: (window[j], j)))
        counts[pattern] = counts.get(pattern, 0) + 1
    p = np.array(list(counts.values()), dtype=np.float64) / n_windows
    h = float(-np.sum(p * np.log(p)))
    if normalize:
        h /= math.log(math.factorial(order))
    return h


class TestPermutationEntropy(unittest.TestCase):
    # ------------------------------------------------------------------
    # Exact-value tests
    # ------------------------------------------------------------------

    def test_monotonic_series_has_zero_entropy(self):
        """Only one ordinal pattern occurs, so entropy is zero."""
        increasing = np.arange(100.0)
        decreasing = np.arange(100.0)[::-1]
        for series in (increasing, decreasing):
            with self.subTest(series=series[:3].tolist()):
                result = permutation_entropy(series, order=3, delay=1)
                self.assertEqual(result.shape, (1,))
                self.assertAlmostEqual(float(result[0]), 0.0, places=12)

    def test_constant_series_has_zero_entropy(self):
        """All values tied: a single pattern, entropy zero."""
        result = permutation_entropy(np.full(200, 3.5), order=4, delay=2)
        self.assertAlmostEqual(float(result[0]), 0.0, places=12)

    def test_exact_small_example_order_two(self):
        """Hand-computed distribution for [1, 3, 2, 4], order=2."""
        series = np.array([1.0, 3.0, 2.0, 4.0])
        # Patterns: up, down, up -> p(up)=2/3, p(down)=1/3.
        expected_raw = -(2.0 / 3.0 * np.log(2.0 / 3.0)
                         + 1.0 / 3.0 * np.log(1.0 / 3.0))
        raw = permutation_entropy(series, order=2, delay=1, normalize=False)
        self.assertAlmostEqual(float(raw[0]), expected_raw, places=12)
        normalized = permutation_entropy(series, order=2, delay=1)
        self.assertAlmostEqual(
            float(normalized[0]), expected_raw / np.log(2.0), places=12
        )

    def test_order_two_balanced_patterns_reaches_one(self):
        """Alternating series visits up/down equally -> maximum entropy."""
        # 201 samples -> 200 windows, exactly 100 up and 100 down.
        series = np.concatenate([np.tile([0.0, 1.0], 100), [0.0]])
        result = permutation_entropy(series, order=2, delay=1)
        self.assertAlmostEqual(float(result[0]), 1.0, places=12)

    # ------------------------------------------------------------------
    # Tie handling
    # ------------------------------------------------------------------

    def test_ties_ordered_by_time_of_appearance(self):
        """Equal values: the earlier position gets the smaller rank."""
        # [1, 1] tie and [1, 2] increasing give the same pattern.
        same = permutation_entropy(
            np.array([1.0, 1.0, 2.0]), order=2, delay=1
        )
        self.assertAlmostEqual(float(same[0]), 0.0, places=12)

        # [2, 1] decreasing and [1, 1] tie (ranked as increasing) differ.
        different = permutation_entropy(
            np.array([2.0, 1.0, 1.0]), order=2, delay=1
        )
        self.assertAlmostEqual(float(different[0]), 1.0, places=12)

    def test_ordinal_codes_tie_rule(self):
        """Direct check of the rank/code helper on tied windows."""
        # Shape (n_windows, n_series, order).
        windows = np.array([
            [[2.0, 1.0, 1.0]],  # ranks (2, 0, 1)
            [[1.0, 1.0, 2.0]],  # ranks (0, 1, 2)
        ])
        codes = _ordinal_codes(windows)
        # Lehmer digits for (2,0,1): two later ranks smaller -> 2*2! = 4.
        # Lehmer digits for (0,1,2): strictly increasing -> 0.
        np.testing.assert_array_equal(codes[:, 0], np.array([4, 0]))

    # ------------------------------------------------------------------
    # Properties and bounds
    # ------------------------------------------------------------------

    def test_normalized_bounds(self):
        """Normalised entropy always lies in [0, 1]."""
        rng = np.random.default_rng(7)
        series = rng.standard_normal(300)
        for order in (2, 3, 4, 5):
            with self.subTest(order=order):
                result = permutation_entropy(series, order=order, delay=1)
                self.assertGreaterEqual(float(result[0]), 0.0)
                self.assertLessEqual(float(result[0]), 1.0)

    def test_raw_entropy_bounded_by_log_factorial(self):
        """Unnormalised entropy is bounded above by ln(order!)."""
        rng = np.random.default_rng(11)
        series = rng.standard_normal(400)
        for order in (2, 3, 4):
            with self.subTest(order=order):
                raw = float(
                    permutation_entropy(
                        series, order=order, normalize=False
                    )[0]
                )
                self.assertGreaterEqual(raw, 0.0)
                self.assertLessEqual(raw, np.log(math.factorial(order)) + 1e-12)

    def test_noise_more_complex_than_periodic(self):
        """White noise must have higher PE than a short repeating pattern."""
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(4000)
        periodic = np.tile(np.array([0.0, 1.0, 2.0, 1.0]), 1000)
        pe_noise = float(permutation_entropy(noise, order=3)[0])
        pe_periodic = float(permutation_entropy(periodic, order=3)[0])
        self.assertGreater(pe_noise, 0.97)
        self.assertLess(pe_periodic, pe_noise)

    def test_deterministic_output(self):
        """Repeated calls return identical results."""
        rng = np.random.default_rng(1)
        series = rng.standard_normal(150)
        first = permutation_entropy(series, order=4, delay=2)
        second = permutation_entropy(series, order=4, delay=2)
        np.testing.assert_array_equal(first, second)

    # ------------------------------------------------------------------
    # delay / order parameters
    # ------------------------------------------------------------------

    def test_delay_is_respected(self):
        """Results with delay > 1 match the brute-force reference."""
        rng = np.random.default_rng(3)
        series = rng.standard_normal(120)
        for order, delay in ((3, 2), (4, 3), (2, 5)):
            with self.subTest(order=order, delay=delay):
                result = float(
                    permutation_entropy(
                        series, order=order, delay=delay
                    )[0]
                )
                expected = _brute_permutation_entropy(
                    series, order, delay, normalize=True
                )
                self.assertAlmostEqual(result, expected, places=12)

    def test_minimal_series_length(self):
        """Exactly (order-1)*delay + 1 samples gives one window -> PE 0."""
        series = np.array([1.0, 5.0, 2.0, 9.0])
        result = permutation_entropy(series, order=4, delay=1)
        self.assertEqual(result.shape, (1,))
        self.assertAlmostEqual(float(result[0]), 0.0, places=12)

    # ------------------------------------------------------------------
    # Brute-force cross checks
    # ------------------------------------------------------------------

    def test_matches_brute_force_univariate(self):
        rng = np.random.default_rng(123)
        series = rng.standard_normal(200)
        for order, delay, normalize in (
            (3, 1, True),
            (4, 2, True),
            (3, 1, False),
            (5, 3, False),
        ):
            with self.subTest(order=order, delay=delay, normalize=normalize):
                result = float(
                    permutation_entropy(
                        series, order=order, delay=delay,
                        normalize=normalize,
                    )[0]
                )
                expected = _brute_permutation_entropy(
                    series, order, delay, normalize
                )
                self.assertAlmostEqual(result, expected, places=12)

    def test_matches_brute_force_with_ties(self):
        """Quantised data exercises the shared tie convention."""
        rng = np.random.default_rng(99)
        series = rng.integers(0, 5, size=300).astype(np.float64)
        for normalize in (True, False):
            with self.subTest(normalize=normalize):
                result = float(
                    permutation_entropy(
                        series, order=4, delay=1, normalize=normalize
                    )[0]
                )
                expected = _brute_permutation_entropy(
                    series, 4, 1, normalize
                )
                self.assertAlmostEqual(result, expected, places=12)

    # ------------------------------------------------------------------
    # Input shapes
    # ------------------------------------------------------------------

    def test_1d_and_equivalent_2d_agree(self):
        rng = np.random.default_rng(5)
        series = rng.standard_normal(180)
        result_1d = permutation_entropy(series, order=3, delay=2)
        result_2d = permutation_entropy(series[:, None], order=3, delay=2)
        self.assertEqual(result_1d.shape, (1,))
        np.testing.assert_allclose(result_1d, result_2d, rtol=0, atol=1e-15)

    def test_list_input_accepted(self):
        series = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8]
        result = permutation_entropy(series, order=3, delay=1)
        expected = _brute_permutation_entropy(
            np.array(series, dtype=np.float64), 3, 1, True
        )
        self.assertAlmostEqual(float(result[0]), expected, places=12)

    def test_multivariate_columns_independent(self):
        """Each column is evaluated independently and returned in order."""
        rng = np.random.default_rng(321)
        col_noise = rng.standard_normal(500)
        col_mono = np.arange(500.0)
        col_periodic = np.tile([0.0, 1.0], 250)
        data = np.column_stack([col_noise, col_mono, col_periodic])

        result = permutation_entropy(data, order=3, delay=1)
        self.assertEqual(result.shape, (3,))
        expected = np.array([
            _brute_permutation_entropy(col_noise, 3, 1, True),
            0.0,
            _brute_permutation_entropy(col_periodic, 3, 1, True),
        ])
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_order_below_two_raises(self):
        for bad in (0, 1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                permutation_entropy(np.arange(50.0), order=bad)

    def test_delay_below_one_raises(self):
        with self.assertRaises(ValueError):
            permutation_entropy(np.arange(50.0), order=3, delay=0)

    def test_series_too_short_raises(self):
        # order=4, delay=2 needs at least 7 samples.
        with self.assertRaises(ValueError):
            permutation_entropy(np.arange(6.0), order=4, delay=2)

    def test_3d_input_raises(self):
        with self.assertRaises(ValueError):
            permutation_entropy(np.zeros((10, 2, 2)), order=3, delay=1)


if __name__ == "__main__":
    unittest.main()
