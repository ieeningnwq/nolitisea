"""Tests for nolitisea.dimension.c2."""

import unittest

import numpy as np

from nolitisea.core.embed import delay_embedding
from nolitisea.dimension.c2 import correlation_integral


def _brute_c2(ts, m, delay, theiler, eps):
    """Brute-force correlation sum for one embedding dimension.

    Counts unique pairs ``(i, j)`` with ``i < j``, ``|j - i| > theiler``
    and Chebyshev distance ``<= eps``, divided by the number of eligible
    pairs.  Mirrors the TISEAN ``c2`` definition independently of the
    cKDTree-based implementation under test.
    """
    E = delay_embedding(ts, m, delay)
    n = E.shape[0]
    count = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(j - i) <= theiler:
                continue
            total += 1
            if np.max(np.abs(E[i] - E[j])) <= eps:
                count += 1
    return count / total if total > 0 else 0.0


class TestC2BruteForce(unittest.TestCase):
    def test_theiler_zero_matches_brute_force(self):
        rng = np.random.default_rng(123)
        s = np.cumsum(rng.standard_normal(120))
        max_emb, delay, theiler, eps = 4, 2, 0, 1.0

        res = correlation_integral(
            s, max_emb=max_emb, delay=delay, theiler=theiler, eps=eps
        )
        for m in range(1, max_emb + 1):
            brute = _brute_c2(s, m, delay, theiler, eps)
            self.assertAlmostEqual(res[m], brute, places=12)

    def test_theiler_positive_matches_brute_force(self):
        rng = np.random.default_rng(7)
        s = np.cumsum(rng.standard_normal(150))
        max_emb, delay, theiler, eps = 3, 1, 3, 0.8

        res = correlation_integral(
            s, max_emb=max_emb, delay=delay, theiler=theiler, eps=eps
        )
        for m in range(1, max_emb + 1):
            brute = _brute_c2(s, m, delay, theiler, eps)
            self.assertAlmostEqual(res[m], brute, places=12)

    def test_large_theiler_matches_brute_force(self):
        rng = np.random.default_rng(99)
        s = rng.standard_normal(80)
        max_emb, delay, theiler, eps = 2, 1, 10, 2.5

        res = correlation_integral(
            s, max_emb=max_emb, delay=delay, theiler=theiler, eps=eps
        )
        for m in range(1, max_emb + 1):
            brute = _brute_c2(s, m, delay, theiler, eps)
            self.assertAlmostEqual(res[m], brute, places=12)


class TestC2Properties(unittest.TestCase):
    def test_c2_monotone_increasing_in_eps(self):
        rng = np.random.default_rng(5)
        s = rng.standard_normal(200)
        eps_values = [0.1, 0.3, 0.6, 1.0, 2.0, 4.0]
        prev = -np.inf
        for eps in eps_values:
            res = correlation_integral(
                s, max_emb=3, delay=1, theiler=1, eps=eps
            )
            # c2 at m=1 must be non-decreasing as eps grows.
            self.assertGreaterEqual(res[1], prev - 1e-12)
            prev = res[1]

    def test_c2_in_range_0_1(self):
        rng = np.random.default_rng(42)
        s = np.cumsum(rng.standard_normal(300))
        for eps in [0.01, 0.5, 2.0, 10.0]:
            res = correlation_integral(
                s, max_emb=5, delay=1, theiler=2, eps=eps
            )
            for c2 in res.values():
                self.assertGreaterEqual(c2, 0.0)
                self.assertLessEqual(c2, 1.0)

    def test_c2_for_large_eps_approaches_one(self):
        rng = np.random.default_rng(11)
        s = rng.standard_normal(100)
        # eps larger than the data range: every eligible pair is counted.
        res = correlation_integral(
            s, max_emb=3, delay=1, theiler=0, eps=1e6
        )
        for c2 in res.values():
            self.assertAlmostEqual(c2, 1.0, places=12)

    def test_c2_for_tiny_eps_is_zero_or_small(self):
        rng = np.random.default_rng(13)
        s = rng.standard_normal(200)
        res = correlation_integral(
            s, max_emb=3, delay=1, theiler=0, eps=1e-10
        )
        for c2 in res.values():
            self.assertLess(c2, 1e-6)

    def test_returns_all_embedding_dimensions(self):
        rng = np.random.default_rng(0)
        s = rng.standard_normal(100)
        max_emb = 6
        res = correlation_integral(
            s, max_emb=max_emb, delay=1, theiler=0, eps=1.0
        )
        self.assertEqual(set(res.keys()), set(range(1, max_emb + 1)))

    def test_default_parameters(self):
        rng = np.random.default_rng(1)
        s = rng.standard_normal(100)
        # Defaults: max_emb=10, delay=1, theiler=0, eps=0.01.
        res = correlation_integral(s)
        self.assertEqual(len(res), 10)
        for c2 in res.values():
            self.assertIsInstance(c2, float)


class TestC2TheilerWindow(unittest.TestCase):
    def test_theiler_reduces_c2(self):
        rng = np.random.default_rng(77)
        s = np.cumsum(rng.standard_normal(200))
        eps = 1.0
        c2_no_theiler = correlation_integral(
            s, max_emb=3, delay=1, theiler=0, eps=eps
        )
        c2_with_theiler = correlation_integral(
            s, max_emb=3, delay=1, theiler=5, eps=eps
        )
        # Excluding temporally close pairs removes near-neighbours that
        # dominate the count at small eps; c2 with theiler should be <=.
        for m in range(1, 4):
            self.assertLessEqual(
                c2_with_theiler[m], c2_no_theiler[m] + 1e-12
            )

    def test_theiler_zero_skips_correction(self):
        rng = np.random.default_rng(33)
        s = rng.standard_normal(100)
        # theiler=0: no subtraction loop; result must match brute force
        # with theiler=0 exactly (no temporal exclusion).
        res = correlation_integral(
            s, max_emb=2, delay=1, theiler=0, eps=1.5
        )
        for m in range(1, 3):
            brute = _brute_c2(s, m, 1, 0, 1.5)
            self.assertAlmostEqual(res[m], brute, places=12)

    def test_theiler_exceeds_n_points(self):
        rng = np.random.default_rng(22)
        s = rng.standard_normal(50)
        # theiler larger than the series leaves no eligible pairs; the
        # function must return 0.0 for every dimension, not crash.
        res = correlation_integral(
            s, max_emb=2, delay=1, theiler=100, eps=1.0
        )
        for c2 in res.values():
            self.assertEqual(c2, 0.0)


class TestC2EdgeCases(unittest.TestCase):
    def test_max_emb_one(self):
        rng = np.random.default_rng(9)
        s = rng.standard_normal(50)
        res = correlation_integral(
            s, max_emb=1, delay=1, theiler=0, eps=1.0
        )
        self.assertEqual(list(res.keys()), [1])
        brute = _brute_c2(s, 1, 1, 0, 1.0)
        self.assertAlmostEqual(res[1], brute, places=12)

    def test_series_too_short_raises(self):
        # ``delay_embedding`` raises ValueError once the embedding dimension
        # exceeds what the series length can support; the caller sees it.
        s = np.arange(3, dtype=float)
        with self.assertRaises(ValueError):
            correlation_integral(
                s, max_emb=10, delay=1, theiler=0, eps=1.0
            )

    def test_integer_input_accepted(self):
        s = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
        res = correlation_integral(
            s, max_emb=3, delay=1, theiler=0, eps=2.0
        )
        # Should not raise; results are valid floats.
        for c2 in res.values():
            self.assertIsInstance(c2, float)
            self.assertGreaterEqual(c2, 0.0)

    def test_list_input_accepted(self):
        res = correlation_integral(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            max_emb=2, delay=1, theiler=0, eps=2.0,
        )
        self.assertEqual(set(res.keys()), {1, 2})


class TestC2SanityChecks(unittest.TestCase):
    def test_uniform_noise_has_higher_c2_at_low_emb(self):
        """Uniform white noise fills space; c2 decreases with m at fixed eps."""
        rng = np.random.default_rng(2024)
        s = rng.standard_normal(500)
        res = correlation_integral(
            s, max_emb=5, delay=1, theiler=2, eps=0.5
        )
        # In higher dimensions fewer pairs fall within a fixed eps ball.
        for m in range(1, 5):
            self.assertLessEqual(
                res[m + 1], res[m] + 1e-12,
                f"c2 should be non-increasing in m at m={m}",
            )

    def test_periodic_signal_high_c2(self):
        """A pure sinusoid embeds onto a 1-D curve; c2 should be high."""
        t = np.arange(400)
        s = np.sin(2.0 * np.pi * t / 50.0)
        res = correlation_integral(
            s, max_emb=3, delay=1, theiler=5, eps=0.1
        )
        # The embedding of a periodic signal is a closed curve; many
        # pairs are close, so c2 should be meaningfully above zero.
        self.assertGreater(res[2], 0.01)


if __name__ == "__main__":
    unittest.main()
