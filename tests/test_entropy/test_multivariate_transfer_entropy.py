"""Tests for nolitisea.entropy.multivariate_transfer_entropy.

The brute-force reference transcribes the conditional TE histogram
estimator with an explicit per-sample dict accumulation, so the
vectorized production code (np.unique packing) can be validated for
exact agreement on small crafted series.
"""

import unittest

import numpy as np

from nolitisea.entropy.multivariate_transfer_entropy import (
    multivariate_transfer_entropy,
)

# ---------------------------------------------------------------------------
# Brute-force reference
# ---------------------------------------------------------------------------


def _brute_te_conditional(source, target, condition, k=1, l=1, l_z: int|list = 1,
                          h: int = 1, bins: int = 10):
    """O(n^2) dict-based histogram conditional TE estimator.

    Builds delay vectors explicitly, digitizes each scalar variable
    with the same equi-width rule as _digitize, and accumulates the
    four required joint distributions in Python dicts.  Computes
    TE = (1/n) * sum_s log2[ n_full * n_c / (n_bc * n_ac) ] per sample
    where A = Y_future, B = X_hist, C = (Y_hist, Z_hist).
    """
    x = np.asarray(source, dtype=np.float64).ravel()
    y = np.asarray(target, dtype=np.float64).ravel()

    # Normalize the conditioning variable(s) into a list of 1-D arrays.
    if isinstance(condition, np.ndarray) and condition.ndim == 2:
        cond_list = [condition[i] for i in range(condition.shape[0])]
    elif isinstance(condition, (list, tuple)):
        cond_list = list(condition)
    else:
        cond_list = [condition]
    cond_list = [np.asarray(c, dtype=np.float64).ravel() for c in cond_list]
    n_cond = len(cond_list)

    if np.ndim(l_z) == 0:
        l_z_vals = [int(l_z)] * n_cond  # pyright: ignore[reportArgumentType]
    else:
        l_z_vals = [int(v) for v in l_z]  # pyright: ignore[reportGeneralTypeIssues]

    N = y.size
    max_hist = max(k - 1, l - 1, max(l_z_vals) - 1)
    n_samples = N - max_hist - h

    y_hist = np.column_stack(
        [y[max_hist - i : max_hist - i + n_samples] for i in range(k)]
    )
    x_hist = np.column_stack(
        [x[max_hist - i : max_hist - i + n_samples] for i in range(l)]
    )
    z_hists = []
    for c_idx in range(n_cond):
        lzi = l_z_vals[c_idx]
        zh = np.column_stack(
            [cond_list[c_idx][max_hist - i : max_hist - i + n_samples]
             for i in range(lzi)]
        )
        z_hists.append(zh)
    y_future = y[max_hist + h : max_hist + h + n_samples]

    def dig(arr):
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.int64)
        edges = np.linspace(lo, hi, bins + 1)[1:-1]
        return np.clip(np.digitize(arr, edges), 0, bins - 1).astype(np.int64)

    y_h_d = [dig(y_hist[:, i]) for i in range(k)]
    x_h_d = [dig(x_hist[:, i]) for i in range(l)]
    z_h_d_all = []
    for c_idx in range(n_cond):
        lzi = l_z_vals[c_idx]
        z_h_d_all.extend(
            [dig(z_hists[c_idx][:, i]) for i in range(lzi)]
        )
    y_f_d = dig(y_future)

    def key_of(indices):
        return tuple(int(i) for i in indices)

    full, bc, ac, c_only = {}, {}, {}, {}
    for s in range(n_samples):
        k_full = key_of([y_f_d[s]] + [y_h_d[j][s] for j in range(k)]
                        + [x_h_d[j][s] for j in range(l)]
                        + [z_h_d_all[j][s] for j in range(len(z_h_d_all))])
        k_bc = key_of([x_h_d[j][s] for j in range(l)]
                      + [y_h_d[j][s] for j in range(k)]
                      + [z_h_d_all[j][s] for j in range(len(z_h_d_all))])
        k_ac = key_of([y_f_d[s]] + [y_h_d[j][s] for j in range(k)]
                      + [z_h_d_all[j][s] for j in range(len(z_h_d_all))])
        k_c = key_of([y_h_d[j][s] for j in range(k)]
                     + [z_h_d_all[j][s] for j in range(len(z_h_d_all))])

        full[k_full] = full.get(k_full, 0) + 1
        bc[k_bc] = bc.get(k_bc, 0) + 1
        ac[k_ac] = ac.get(k_ac, 0) + 1
        c_only[k_c] = c_only.get(k_c, 0) + 1

    te = 0.0
    for s in range(n_samples):
        k_full = key_of([y_f_d[s]] + [y_h_d[j][s] for j in range(k)]
                        + [x_h_d[j][s] for j in range(l)]
                        + [z_h_d_all[j][s] for j in range(len(z_h_d_all))])
        k_bc = key_of([x_h_d[j][s] for j in range(l)]
                      + [y_h_d[j][s] for j in range(k)]
                      + [z_h_d_all[j][s] for j in range(len(z_h_d_all))])
        k_ac = key_of([y_f_d[s]] + [y_h_d[j][s] for j in range(k)]
                      + [z_h_d_all[j][s] for j in range(len(z_h_d_all))])
        k_c = key_of([y_h_d[j][s] for j in range(k)]
                     + [z_h_d_all[j][s] for j in range(len(z_h_d_all))])
        ratio = (full[k_full] * c_only[k_c]
                 / (bc[k_bc] * ac[k_ac]))
        te += np.log2(ratio)
    return te / n_samples


# ---------------------------------------------------------------------------
# Brute-force agreement
# ---------------------------------------------------------------------------


class TestMultivariateTEBruteForce(unittest.TestCase):
    """Compare vectorized conditional TE against the dict-based reference."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.x = rng.standard_normal(80)
        cls.y = cls.x * 0.7 + 0.3 * rng.standard_normal(80)
        cls.z = rng.standard_normal(80)
        cls.x_long = rng.standard_normal(200)
        cls.y_long = rng.standard_normal(200)
        cls.z_long = rng.standard_normal(200)

    def test_default_params(self):
        got = multivariate_transfer_entropy(self.x, self.y, self.z)
        ref = _brute_te_conditional(self.x, self.y, self.z)
        self.assertAlmostEqual(got, ref, places=12)

    def test_bins_variation(self):
        for bins in [2, 3, 5, 8]:
            with self.subTest(bins=bins):
                got = multivariate_transfer_entropy(
                    self.x, self.y, self.z, bins=bins
                )
                ref = _brute_te_conditional(
                    self.x, self.y, self.z, bins=bins
                )
                self.assertAlmostEqual(got, ref, places=12)

    def test_k_l_lz_h_variation(self):
        for k, l, lz, h in [(1, 1, 1, 1), (2, 1, 1, 1), (1, 2, 1, 1),
                            (1, 1, 2, 1), (1, 1, 1, 2), (2, 2, 2, 1),
                            (2, 1, 2, 2), (3, 2, 1, 1), (2, 3, 2, 1)]:
            with self.subTest(k=k, l=l, l_z=lz, h=h):
                got = multivariate_transfer_entropy(
                    self.x_long, self.y_long, self.z_long,
                    k=k, l=l, l_z=lz, h=h, bins=4,
                )
                ref = _brute_te_conditional(
                    self.x_long, self.y_long, self.z_long,
                    k=k, l=l, l_z=lz, h=h, bins=4,
                )
                self.assertAlmostEqual(got, ref, places=12)

    def test_multiple_conditioning_variables(self):
        rng = np.random.default_rng(7)
        z1 = rng.standard_normal(150)
        z2 = rng.standard_normal(150)
        z3 = rng.standard_normal(150)
        x = rng.standard_normal(150)
        y = 0.5 * x + 0.3 * z1 + 0.2 * z2 + 0.1 * z3 \
            + 0.3 * rng.standard_normal(150)
        got = multivariate_transfer_entropy(x, y, [z1, z2, z3], bins=4)
        ref = _brute_te_conditional(x, y, [z1, z2, z3], bins=4)
        self.assertAlmostEqual(got, ref, places=12)

    def test_condition_as_2d_array(self):
        rng = np.random.default_rng(5)
        z2d = rng.standard_normal((2, 100))
        x = rng.standard_normal(100)
        y = 0.4 * x + 0.3 * z2d[0] + 0.2 * z2d[1] \
            + 0.3 * rng.standard_normal(100)
        got = multivariate_transfer_entropy(x, y, z2d, bins=4)
        ref = _brute_te_conditional(x, y, z2d, bins=4)
        self.assertAlmostEqual(got, ref, places=12)

    def test_per_variable_lz_sequence(self):
        rng = np.random.default_rng(3)
        z1 = rng.standard_normal(150)
        z2 = rng.standard_normal(150)
        x = rng.standard_normal(150)
        y = 0.5 * x + 0.3 * z1 + 0.2 * z2 + 0.3 * rng.standard_normal(150)
        got = multivariate_transfer_entropy(
            x, y, [z1, z2], l_z=[2, 3], bins=4  # pyright: ignore[reportArgumentType]
        )
        ref = _brute_te_conditional(
            x, y, [z1, z2], l_z=[2, 3], bins=4
        )
        self.assertAlmostEqual(got, ref, places=12)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestMultivariateTEEdgeCases(unittest.TestCase):
    """Edge-case behavior of the conditional TE estimator."""

    def test_constant_source(self):
        # Constant source collapses to a single bin -> TE = 0 (no
        # information beyond target's past and Z).
        x = np.full(50, 2.0)
        y = np.linspace(0, 10, 50)
        z = np.linspace(0, 5, 50)
        got = multivariate_transfer_entropy(x, y, z, bins=5)
        ref = _brute_te_conditional(x, y, z, bins=5)
        self.assertAlmostEqual(got, ref, places=12)
        self.assertEqual(got, 0.0)

    def test_constant_target(self):
        x = np.linspace(0, 10, 50)
        y = np.full(50, 2.0)
        z = np.linspace(0, 5, 50)
        got = multivariate_transfer_entropy(x, y, z, bins=5)
        ref = _brute_te_conditional(x, y, z, bins=5)
        self.assertAlmostEqual(got, ref, places=12)
        self.assertEqual(got, 0.0)

    def test_constant_condition(self):
        # Constant condition collapses to a single bin; TE(X->Y|Z=const)
        # should equal the bivariate TE(X->Y) since the constant carries
        # no extra information.
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        y = 0.5 * x + 0.5 * rng.standard_normal(200)
        z = np.full(200, 1.0)
        from nolitisea.entropy.transfer_entropy import transfer_entropy
        te_cond = multivariate_transfer_entropy(x, y, z, bins=6)
        te_biv = transfer_entropy(x, y, bins=6)
        # The constant Z adds no information, so the conditional MI
        # collapses to the unconditional MI (extra bin does not change
        # the count since every sample shares the same Z value).
        self.assertAlmostEqual(te_cond, te_biv, places=12)

    def test_identical_source_and_condition(self):
        # If X == Z, conditioning on Z should remove the X contribution
        # (once Z is known, X is fully determined and adds no info).
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        y = 0.5 * x + 0.5 * rng.standard_normal(200)
        te_cond = multivariate_transfer_entropy(x, y, x, bins=6)
        self.assertEqual(te_cond, 0.0)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestMultivariateTEValidation(unittest.TestCase):
    """Tests for parameter validation."""

    def test_source_target_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            multivariate_transfer_entropy(
                np.arange(10.0), np.arange(11.0), np.arange(10.0)
            )

    def test_condition_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            multivariate_transfer_entropy(
                np.arange(20.0), np.arange(20.0), np.arange(19.0)
            )

    def test_one_condition_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            multivariate_transfer_entropy(
                np.arange(20.0), np.arange(20.0),
                [np.arange(20.0), np.arange(19.0)],
            )

    def test_invalid_k_raises(self):
        with self.assertRaises(ValueError):
            multivariate_transfer_entropy(
                np.arange(20.0), np.arange(20.0), np.arange(20.0), k=0
            )

    def test_invalid_l_raises(self):
        with self.assertRaises(ValueError):
            multivariate_transfer_entropy(
                np.arange(20.0), np.arange(20.0), np.arange(20.0), l=0
            )

    def test_invalid_lz_int_raises(self):
        with self.assertRaises(ValueError):
            multivariate_transfer_entropy(
                np.arange(20.0), np.arange(20.0), np.arange(20.0), l_z=0
            )

    def test_invalid_lz_sequence_entry_raises(self):
        with self.assertRaises(ValueError):
            multivariate_transfer_entropy(
                np.arange(20.0), np.arange(20.0),
                [np.arange(20.0), np.arange(20.0)],
                l_z=[1, 0],  # pyright: ignore[reportArgumentType]
            )

    def test_lz_sequence_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            multivariate_transfer_entropy(
                np.arange(20.0), np.arange(20.0),
                [np.arange(20.0), np.arange(20.0)],
                l_z=[1, 1, 1],  # 3 entries but 2 condition vars  # pyright: ignore[reportArgumentType]
            )

    def test_invalid_h_raises(self):
        with self.assertRaises(ValueError):
            multivariate_transfer_entropy(
                np.arange(20.0), np.arange(20.0), np.arange(20.0), h=0
            )

    def test_invalid_bins_raises(self):
        with self.assertRaises(ValueError):
            multivariate_transfer_entropy(
                np.arange(20.0), np.arange(20.0), np.arange(20.0), bins=0
            )

    def test_too_short_series_raises(self):
        # N=4, k=2, l=2, lz=2, h=1 -> max_hist=1, n_samples = 4-1-1 = 2 (ok)
        # N=4, k=2, l=2, lz=2, h=2 -> n_samples = 4-1-2 = 1 (ok)
        # N=4, k=2, l=2, lz=2, h=3 -> n_samples = 4-1-3 = 0 (raises)
        with self.assertRaises(ValueError):
            multivariate_transfer_entropy(
                np.arange(4.0), np.arange(4.0), np.arange(4.0),
                k=2, l=2, l_z=2, h=3,
            )

    def test_2d_source_accepted(self):
        # 2-D source/target/condition are raveled to 1-D.  Pass the
        # condition as a list with a single 2-D entry so it is raveled
        # rather than split into rows (which a bare 2-D array would be).
        x = np.arange(20.0).reshape(2, 10)
        y = np.arange(20.0).reshape(2, 10)
        z = np.arange(20.0).reshape(2, 10)
        # All ravel to [0..19] of equal length -> valid.
        te = multivariate_transfer_entropy(x, y, [z], bins=4)
        self.assertIsInstance(te, float)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestMultivariateTEProperties(unittest.TestCase):
    """Tests for non-negativity, determinism, and basic properties."""

    def test_returns_float(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(100)
        y = rng.standard_normal(100)
        z = rng.standard_normal(100)
        self.assertIsInstance(
            multivariate_transfer_entropy(x, y, z, bins=5), float
        )

    def test_non_negative_on_coupled_data(self):
        rng = np.random.default_rng(0)
        n = 500
        x = np.zeros(n)
        y = np.zeros(n)
        z = np.zeros(n)
        for t in range(n - 1):
            z[t + 1] = 0.7 * z[t] + 0.3 * rng.standard_normal()
            x[t + 1] = 0.5 * x[t] + 0.3 * z[t] + 0.2 * rng.standard_normal()
            y[t + 1] = 0.4 * y[t] + 0.5 * x[t] + 0.2 * rng.standard_normal()
        te = multivariate_transfer_entropy(x, y, z, bins=8)
        self.assertGreaterEqual(te, 0.0)

    def test_deterministic_output(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        y = 0.5 * x + 0.5 * rng.standard_normal(200)
        z = rng.standard_normal(200)
        te1 = multivariate_transfer_entropy(x, y, z, bins=6)
        te2 = multivariate_transfer_entropy(x, y, z, bins=6)
        self.assertEqual(te1, te2)

    def test_independent_series_near_zero(self):
        # Three independent white-noise series: conditional TE small.
        # With 4 conditioning variables (y_h, x_h, z_h, y_f) at 4 bins
        # each the joint space is 256 cells; finite-sample bias is
        # larger than the bivariate case, so use a relaxed threshold.
        rng = np.random.default_rng(0)
        x = rng.standard_normal(8000)
        y = rng.standard_normal(8000)
        z = rng.standard_normal(8000)
        te = multivariate_transfer_entropy(x, y, z, bins=4)
        self.assertLess(abs(te), 0.05)


# ---------------------------------------------------------------------------
# Conditioning effect: common-cause removal vs. direct link survives
# ---------------------------------------------------------------------------


class TestConditioningEffect(unittest.TestCase):
    """Verify conditioning on Z removes spurious common-cause TE."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        n = 8000
        # Case 1: common cause.  Z autocorrelated drives both X and Y.
        z = np.zeros(n)
        x = np.zeros(n)
        y = np.zeros(n)
        z[0] = rng.standard_normal()
        for t in range(n - 1):
            z[t + 1] = 0.95 * z[t] + 0.1 * rng.standard_normal()
            x[t + 1] = z[t] + 0.02 * rng.standard_normal()
            y[t + 1] = z[t] + 0.02 * rng.standard_normal()
        cls.z_common = z
        cls.x_common = x
        cls.y_common = y

        # Case 2: direct link.  X -> Y directly, Z also drives Y.
        x2 = rng.standard_normal(n)
        z2 = rng.standard_normal(n)
        y2 = np.zeros(n)
        for t in range(n - 1):
            y2[t + 1] = (0.5 * y2[t] + 0.5 * x2[t] + 0.3 * z2[t]
                         + 0.2 * rng.standard_normal())
        cls.x_direct = x2
        cls.y_direct = y2
        cls.z_direct = z2

    def test_common_cause_conditioning_reduces_te(self):
        from nolitisea.entropy.transfer_entropy import transfer_entropy
        te_biv = transfer_entropy(self.x_common, self.y_common, bins=8)
        te_cond = multivariate_transfer_entropy(
            self.x_common, self.y_common, self.z_common, bins=8
        )
        # Conditioning on the common driver should reduce TE substantially.
        self.assertLess(te_cond, te_biv)
        # And the conditional TE should be close to zero (within finite-sample
        # noise).
        self.assertLess(te_cond, 0.02)

    def test_direct_link_survives_conditioning(self):
        te_cond = multivariate_transfer_entropy(
            self.x_direct, self.y_direct, self.z_direct, bins=8
        )
        # Direct X->Y link should remain clearly positive after conditioning.
        self.assertGreater(te_cond, 0.1)

    def test_conditioning_removes_spurious_more_than_direct(self):
        # The common-cause case should show a larger relative drop than
        # the direct-link case.
        from nolitisea.entropy.transfer_entropy import transfer_entropy
        te_biv_common = transfer_entropy(
            self.x_common, self.y_common, bins=8
        )
        te_cond_common = multivariate_transfer_entropy(
            self.x_common, self.y_common, self.z_common, bins=8
        )
        drop_common = te_biv_common - te_cond_common

        te_biv_direct = transfer_entropy(
            self.x_direct, self.y_direct, bins=8
        )
        te_cond_direct = multivariate_transfer_entropy(
            self.x_direct, self.y_direct, self.z_direct, bins=8
        )
        drop_direct = te_biv_direct - te_cond_direct
        # Spurious link should be removed more than the direct link.
        self.assertGreater(drop_common, drop_direct)

    def test_multiple_conditioning_variables(self):
        # Two conditioning variables; should run without crash and
        # produce a finite result.
        rng = np.random.default_rng(0)
        n = 3000
        z1 = rng.standard_normal(n)
        z2 = rng.standard_normal(n)
        x = rng.standard_normal(n)
        y_arr = np.zeros(n)
        for t in range(n - 1):
            y_arr[t + 1] = (0.4 * y_arr[t] + 0.3 * z1[t] + 0.3 * z2[t]
                            + 0.2 * rng.standard_normal())
        te = multivariate_transfer_entropy(
            x, y_arr, [z1, z2], bins=6
        )
        self.assertTrue(np.isfinite(te))


if __name__ == "__main__":
    unittest.main()
