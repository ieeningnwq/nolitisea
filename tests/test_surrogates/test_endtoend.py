"""Tests for ``nolitisea.surrogates.endtoend``.

The mismatch metric only inspects the wrap-around seam (last/first
values and the first finite difference across it), so the constructed
cases place their discontinuities at the seam — an interior spike is
invisible to the algorithm by design.
"""

import unittest

import numpy as np

from nolitisea.surrogates.endtoend import endtoend

# ---------------------------------------------------------------------------
# Brute-force reference (direct transcription of the definition)
# ---------------------------------------------------------------------------

def _brute_endtoend(data, wjump):
    """O(n^2)-per-length reference: scan every (L, offset) explicitly."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    n, n_vars = data.shape

    best_etot = np.inf
    best = (n, 0, 0.0, 0.0)
    for L in range(n, 2, -1):
        n_win = n - L + 1
        for off in range(n_win):
            xj = 0.0
            sj = 0.0
            for c in range(n_vars):
                col = data[off : off + L, c]
                sd = np.std(col)  # population, matches _sliding_pop_std
                denom = L * sd * sd
                if denom > 0:
                    xj += (col[0] - col[-1]) ** 2 / denom
                    sj += ((col[-1] - col[-2]) - (col[1] - col[0])) ** 2 / denom
            etot = wjump * xj + (1.0 - wjump) * sj
            if etot < best_etot:
                best_etot = etot
                best = (L, off, xj, sj)
        if best_etot < 1e-5:
            break
    return best  # (length, offset, jump, slip)


def _random_walk(n, seed, n_vars=1):
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal((n, n_vars))
    return np.cumsum(steps, axis=0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation(unittest.TestCase):
    def test_too_short_series(self):
        with self.assertRaises(ValueError):
            endtoend(np.array([1.0]))
        with self.assertRaises(ValueError):
            endtoend(np.array([1.0, 2.0]))

    def test_three_dimensional_input(self):
        with self.assertRaises(ValueError):
            endtoend(np.zeros((4, 2, 2)))

    def test_wjump_below_range(self):
        with self.assertRaises(ValueError):
            endtoend(np.arange(10.0), wjump=-0.1)

    def test_wjump_above_range(self):
        with self.assertRaises(ValueError):
            endtoend(np.arange(10.0), wjump=1.1)

    def test_wjump_bounds_accepted(self):
        x = np.arange(10.0)
        for wj in (0.0, 1.0):
            res = endtoend(x, wjump=wj)
            self.assertIsInstance(res, dict)

    def test_list_input_accepted(self):
        res = endtoend([0.0, 1.0, 2.0, 3.0, 2.0, 1.0])
        self.assertEqual(res["length"], 6)


# ---------------------------------------------------------------------------
# Hand-computed cases
# ---------------------------------------------------------------------------

class TestHandComputed(unittest.TestCase):
    def test_constant_series_full_length(self):
        # Every window has sd == 0 -> zero mismatch everywhere; the scan
        # stops immediately at the full length (also exercises the
        # zero-denominator guard and the early break).
        res = endtoend(np.full(12, 5.0))
        self.assertEqual(res["length"], 12)
        self.assertEqual(res["offset"], 0)
        self.assertEqual(res["lost"], 0.0)
        self.assertEqual(res["jump"], 0.0)
        self.assertEqual(res["slip"], 0.0)
        self.assertEqual(res["weighted"], 0.0)

    def test_step_series_picks_longest_plateau(self):
        # 4 zeros followed by 6 tens: the first constant window appears
        # at L = 6 (the 6-sample plateau), offset 4.
        x = np.concatenate([np.zeros(4), np.full(6, 10.0)])
        res = endtoend(x)
        self.assertEqual(res["length"], 6)
        self.assertEqual(res["offset"], 4)
        self.assertAlmostEqual(res["lost"], 0.4, places=12)
        self.assertEqual(res["jump"], 0.0)
        self.assertEqual(res["slip"], 0.0)
        self.assertEqual(res["weighted"], 0.0)

    def test_leading_spike_is_cut_off(self):
        # A spike at the very first sample creates a large seam
        # discontinuity for every window containing offset 0; the first
        # seam-free window is the 11-sample tail (offset 1).
        x = np.concatenate([[1000.0], np.zeros(11)])
        res = endtoend(x)
        self.assertEqual(res["length"], 11)
        self.assertEqual(res["offset"], 1)
        self.assertAlmostEqual(res["lost"], 1.0 / 12.0, places=12)
        self.assertEqual(res["jump"], 0.0)
        self.assertEqual(res["slip"], 0.0)
        self.assertEqual(res["weighted"], 0.0)

    def test_interior_spike_is_invisible(self):
        # The metric only measures the wrap-around seam, so a spike in
        # the middle does not shorten the slice at all.
        x = np.concatenate([np.zeros(8), [1000.0], np.zeros(3)])
        res = endtoend(x)
        self.assertEqual(res["length"], 12)
        self.assertEqual(res["offset"], 0)

    def test_periodic_sine_keeps_full_length(self):
        i = np.arange(40)
        x = np.sin(2.0 * np.pi * i / 40)
        res = endtoend(x)
        self.assertEqual(res["length"], 40)
        self.assertEqual(res["offset"], 0)
        self.assertLess(res["weighted"], 0.01)

    def test_lost_fraction_consistent(self):
        x = np.concatenate([np.zeros(4), np.full(6, 10.0)])
        res = endtoend(x)
        self.assertAlmostEqual(
            res["lost"], (10 - res["length"]) / 10.0, places=12
        )


# ---------------------------------------------------------------------------
# wjump semantics
# ---------------------------------------------------------------------------

class TestWjump(unittest.TestCase):
    def test_weighted_identity(self):
        x = _random_walk(40, seed=0)[:, 0]
        for wj in (0.0, 0.25, 0.5, 1.0):
            res = endtoend(x, wjump=wj)
            self.assertAlmostEqual(
                res["weighted"],
                wj * res["jump"] + (1.0 - wj) * res["slip"],
                places=12,
            )

    def test_wjump_one_returns_jump_only(self):
        x = _random_walk(40, seed=0)[:, 0]
        res = endtoend(x, wjump=1.0)
        self.assertEqual(res["weighted"], res["jump"])

    def test_wjump_zero_returns_slip_only(self):
        x = _random_walk(40, seed=0)[:, 0]
        res = endtoend(x, wjump=0.0)
        self.assertEqual(res["weighted"], res["slip"])


# ---------------------------------------------------------------------------
# Brute-force cross-check
# ---------------------------------------------------------------------------

class TestBruteForceAgreement(unittest.TestCase):
    def test_1d_random_walk(self):
        for seed in (0, 1):
            x = _random_walk(40, seed=seed)[:, 0]
            for wj in (0.0, 0.3, 0.5, 1.0):
                with self.subTest(seed=seed, wjump=wj):
                    res = endtoend(x, wjump=wj)
                    L, off, xj, sj = _brute_endtoend(x, wj)
                    self.assertEqual(res["length"], L)
                    self.assertEqual(res["offset"], off)
                    self.assertAlmostEqual(res["jump"], xj, places=10)
                    self.assertAlmostEqual(res["slip"], sj, places=10)
                    self.assertAlmostEqual(
                        res["weighted"],
                        wj * xj + (1.0 - wj) * sj,
                        places=10,
                    )

    def test_2d_random_walk(self):
        x = _random_walk(30, seed=2, n_vars=3)
        for wj in (0.0, 0.5, 1.0):
            with self.subTest(wjump=wj):
                res = endtoend(x, wjump=wj)
                L, off, xj, sj = _brute_endtoend(x, wj)
                self.assertEqual(res["length"], L)
                self.assertEqual(res["offset"], off)
                self.assertAlmostEqual(res["jump"], xj, places=10)
                self.assertAlmostEqual(res["slip"], sj, places=10)

    def test_smooth_series(self):
        i = np.arange(50)
        x = np.sin(2.0 * np.pi * i / 50) + 0.05 * np.sin(2.0 * np.pi * 3 * i / 50)
        res = endtoend(x, wjump=0.5)
        L, off, xj, sj = _brute_endtoend(x, 0.5)
        self.assertEqual(res["length"], L)
        self.assertEqual(res["offset"], off)
        self.assertAlmostEqual(res["weighted"], 0.5 * xj + 0.5 * sj, places=10)


# ---------------------------------------------------------------------------
# Multivariate handling
# ---------------------------------------------------------------------------

class TestMultivariate(unittest.TestCase):
    def test_bad_seam_in_one_component_shortens_slice(self):
        i = np.arange(40)
        good = np.sin(2.0 * np.pi * i / 40)[:, None]
        bad = np.concatenate([[1000.0], np.zeros(39)])[:, None]
        r_good = endtoend(good)
        r_two = endtoend(np.hstack([good, bad]))
        self.assertEqual(r_good["length"], 40)
        self.assertLess(r_two["length"], 40)

    def test_1d_and_column_input_identical(self):
        x = _random_walk(30, seed=4)[:, 0]
        self.assertEqual(endtoend(x), endtoend(x[:, None]))

    def test_offset_slicing_matches_window(self):
        # The reported (length, offset) window must itself have (near-)
        # zero seam mismatch: re-running endtoend on the slice keeps the
        # full length.
        x = _random_walk(40, seed=1)[:, 0]
        res = endtoend(x)
        window = x[res["offset"] : res["offset"] + res["length"]]
        res2 = endtoend(window)
        self.assertEqual(res2["length"], window.size)
        self.assertLessEqual(res2["weighted"], res["weighted"] + 1e-12)


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

class TestReturnTypes(unittest.TestCase):
    def test_keys_and_types(self):
        res = endtoend(_random_walk(25, seed=5)[:, 0])
        self.assertEqual(
            set(res),
            {"length", "offset", "lost", "jump", "slip", "weighted"},
        )
        self.assertIsInstance(res["length"], int)
        self.assertIsInstance(res["offset"], int)
        for key in ("lost", "jump", "slip", "weighted"):
            self.assertIsInstance(res[key], float)

    def test_mismatch_values_nonnegative(self):
        res = endtoend(_random_walk(25, seed=5)[:, 0])
        for key in ("jump", "slip", "weighted"):
            self.assertGreaterEqual(res[key], 0.0)


if __name__ == "__main__":
    unittest.main()
