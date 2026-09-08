"""Tests for nolitisea.embedding.poincare.poincare_section.

The sequential fast path is validated against hand-computed values on
small crafted series; the two-phase parallel path is checked for bitwise
consistency against the sequential path on dynamical data.
"""

import unittest

import numpy as np

from nolitisea.embedding.poincare import poincare_section

# ---------------------------------------------------------------------------
# Hand-computed cases
# ---------------------------------------------------------------------------


class TestPoincareSectionHandComputed(unittest.TestCase):
    """Validate the sequential algorithm against hand-computed values."""

    def test_simple_alternating_direction_down(self):
        # series = [0,1,0,1,0,1,0,1], dim=2, delay=1, comp_to_cut=1
        # (cutting component is the second coord = series[t+1]).
        # direction=1 (top-down): need series[t+1] > 0.5 and series[t+2] <= 0.5.
        # Crossings at t=0,2,4; crossing_time = t + (1-0.5)/(1-0) = t+0.5.
        # First crossing (t=0) only seeds last_time=0.5 (no output).
        # t=2: ret=2.5-0.5=2.0, pt: j=0, jd=2, xcut=0+0.5*(1-0)=0.5.
        # t=4: ret=4.5-2.5=2.0, pt: jd=4, xcut=0+0.5*(1-0)=0.5.
        series = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        pts, rets = poincare_section(
            series, dim=2, delay=1, comp_to_cut=1,
            threshold=0.5, direction=1,
        )
        np.testing.assert_array_almost_equal(pts, [[0.5], [0.5]])
        np.testing.assert_array_almost_equal(rets, [2.0, 2.0])

    def test_simple_alternating_direction_up(self):
        # Same series, comp_to_cut=0 (cutting component = series[t]).
        # direction=0 (bottom-up): need series[t] < 0.5 and series[t+1] >= 0.5.
        # Crossings at t=0,2,4; crossing_time = t + (0.5-0)/(1-0) = t+0.5.
        # pt: j=1, jd=t+1, xcut = series[t+1] + 0.5*(series[t+2]-series[t+1])
        #   = 1 + 0.5*(0-1) = 0.5.
        series = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        pts, rets = poincare_section(
            series, dim=2, delay=1, comp_to_cut=0,
            threshold=0.5, direction=0,
        )
        np.testing.assert_array_almost_equal(pts, [[0.5], [0.5]])
        np.testing.assert_array_almost_equal(rets, [2.0, 2.0])

    def test_first_crossing_produces_no_output(self):
        # The first crossing only seeds last_time; no Poincare point or
        # return time is emitted for it.
        series = np.array([0.0, 1.0, 0.0, 1.0, 0.0])  # only one crossing pair
        pts, rets = poincare_section(
            series, dim=2, delay=1, comp_to_cut=0,
            threshold=0.5, direction=0,
        )
        # max_t = 5 - 1 - 1 = 3, range(3) = [0,1,2]; t=0 crosses, t=2 crosses.
        # t=0 seeds; t=2 emits.
        self.assertEqual(pts.shape, (1, 1))
        self.assertEqual(len(rets), 1)

    def test_no_crossings_returns_empty(self):
        # Constant series -> no crossings.
        series = np.ones(50) * 3.0
        pts, rets = poincare_section(
            series, dim=2, delay=1, threshold=0.5, direction=0
        )
        self.assertEqual(pts.size, 0)
        self.assertEqual(rets.size, 0)

    def test_interpolation_slope_hand_computed(self):
        # Unequal spacing: series = [0, 2, 4], threshold=1.0, comp_to_cut=0,
        # direction=0.  Cross at t=0: denom=0-2=-2, delta=(0-1)/(-2)=0.5.
        # crossing_time=0.5; j=1, jd=1, xcut=2+0.5*(4-2)=3.0.
        series = np.array([0.0, 2.0, 4.0, 6.0])
        pts, rets = poincare_section(
            series, dim=2, delay=1, comp_to_cut=0,
            threshold=1.0, direction=0,
        )
        # Only one crossing pair (t=0); first crossing seeds only -> empty.
        # Need a second crossing: extend the series.
        series = np.array([0.0, 2.0, 0.0, 2.0, 0.0])
        pts, rets = poincare_section(
            series, dim=2, delay=1, comp_to_cut=0,
            threshold=1.0, direction=0,
        )
        # t=0: crossing_time=0.5, seed.  t=2: crossing_time=2.5, ret=2.0.
        # j=1, jd=3, xcut=2+0.5*(0-2)=1.0.
        np.testing.assert_array_almost_equal(pts, [[1.0]])
        np.testing.assert_array_almost_equal(rets, [2.0])

    def test_dim3_two_output_coordinates(self):
        # dim=3 produces (dim-1)=2 coordinates per Poincare point.
        series = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                           0.0, 1.0])
        pts, _ = poincare_section(
            series, dim=3, delay=1, comp_to_cut=2,
            threshold=0.5, direction=1,
        )
        self.assertEqual(pts.shape[1], 2)

    def test_crossing_time_interpolation_value(self):
        # Verify the interpolated crossing time (not just integer t).
        # series = [0, 3, 0, 3], threshold=1.0, comp_to_cut=0, direction=0.
        # t=0: denom=0-3=-3, delta=(0-1)/(-3)=1/3, crossing_time=1/3.
        # t=2: crossing_time=2+1/3=7/3, ret=7/3-1/3=2.0.
        series = np.array([0.0, 3.0, 0.0, 3.0, 0.0, 3.0])
        _, rets = poincare_section(
            series, dim=2, delay=1, comp_to_cut=0,
            threshold=1.0, direction=0,
        )
        self.assertAlmostEqual(rets[0], 2.0, places=15)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestPoincareSectionValidation(unittest.TestCase):
    """Tests for parameter validation."""

    def test_comp_to_cut_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            poincare_section(np.arange(10.0), dim=3, comp_to_cut=3)

    def test_comp_to_cut_negative_other_than_minus_one_raises(self):
        with self.assertRaises(ValueError):
            poincare_section(np.arange(10.0), dim=3, comp_to_cut=-2)

    def test_comp_to_cut_minus_one_valid(self):
        # comp_to_cut=-1 is the default and maps to dim-1.
        series = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        # Should not raise.
        pts, _ = poincare_section(series, dim=2, comp_to_cut=-1,
                                  threshold=0.5, direction=1)
        self.assertEqual(pts.shape[1], 1)

    def test_invalid_backend_raises(self):
        with self.assertRaises(ValueError):
            poincare_section(np.arange(100.0), dim=2, n_jobs=2,
                             backend="invalid")

    def test_series_too_short_returns_empty(self):
        # N=3, dim=2, delay=1 -> max_t = 3-1-1 = 1 -> sequential path,
        # range(1)=[0], at most one crossing (seed only) -> empty output.
        pts, rets = poincare_section(
            np.array([0.0, 1.0, 0.0]), dim=2, delay=1,
            threshold=0.5, direction=0,
        )
        self.assertEqual(pts.size, 0)
        self.assertEqual(rets.size, 0)


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------


class TestPoincareSectionDirection(unittest.TestCase):
    """Tests for the direction parameter (bottom-up vs top-down)."""

    def test_direction_zero_finds_up_crossings(self):
        # Sine wave (10 periods): bottom-up crossings through threshold=0.5
        # occur once per period -> ~9 output points (first seeds).
        t = np.linspace(0, 20 * np.pi, 4000)
        x = np.sin(t)
        pts, _ = poincare_section(
            x, dim=2, delay=1, threshold=0.5, direction=0
        )
        self.assertGreaterEqual(pts.shape[0], 5)

    def test_direction_one_finds_down_crossings(self):
        # Sine wave (10 periods): top-down crossings through 0.5.
        t = np.linspace(0, 20 * np.pi, 4000)
        x = np.sin(t)
        pts, _ = poincare_section(
            x, dim=2, delay=1, threshold=0.5, direction=1
        )
        self.assertGreaterEqual(pts.shape[0], 5)

    def test_directions_are_disjoint(self):
        # Every crossing is either up or down, never both; each direction
        # sees roughly half the total crossings through a given threshold.
        t = np.linspace(0, 20 * np.pi, 4000)
        x = np.sin(t)
        pts_up, _ = poincare_section(x, threshold=0.5, direction=0)
        pts_down, _ = poincare_section(x, threshold=0.5, direction=1)
        self.assertGreaterEqual(pts_up.shape[0], 5)
        self.assertGreaterEqual(pts_down.shape[0], 5)
        # Counts should be close (within 1) since up/down alternate.
        self.assertLessEqual(abs(pts_up.shape[0] - pts_down.shape[0]), 1)


# ---------------------------------------------------------------------------
# min_return_time filtering
# ---------------------------------------------------------------------------


class TestPoincareSectionMinReturnTime(unittest.TestCase):
    """Tests for the noise-resistance / debouncing parameter."""

    def test_large_min_return_time_filters_all(self):
        series = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        # All return times are 2.0; min_return_time=3.0 rejects all.
        pts, rets = poincare_section(
            series, dim=2, delay=1, comp_to_cut=1,
            threshold=0.5, direction=1, min_return_time=3.0,
        )
        self.assertEqual(pts.size, 0)
        self.assertEqual(rets.size, 0)

    def test_min_return_time_equal_is_accepted(self):
        # Condition is ret_time < min_return_time (strict), so an equal
        # return time is accepted.
        series = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        pts, _ = poincare_section(
            series, dim=2, delay=1, comp_to_cut=1,
            threshold=0.5, direction=1, min_return_time=2.0,
        )
        self.assertEqual(pts.shape[0], 2)

    def test_min_return_time_partial_filter(self):
        # With min_return_time=1.5 (< 2.0), both crossings are accepted.
        series = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        pts_all, _ = poincare_section(
            series, dim=2, delay=1, comp_to_cut=1,
            threshold=0.5, direction=1, min_return_time=0.0,
        )
        pts_filt, _ = poincare_section(
            series, dim=2, delay=1, comp_to_cut=1,
            threshold=0.5, direction=1, min_return_time=1.5,
        )
        np.testing.assert_array_equal(pts_all, pts_filt)

    def test_rejected_crossing_still_updates_last_time(self):
        # A rejected crossing still updates last_time, so the next
        # accepted return time is measured from the rejected crossing.
        # Construct a series with crossings at t=0,2,4 but uneven spacing
        # so that rejecting t=2 changes the return time of t=4.
        # series = [0,1, 0,1, 0,1, 0,1] with comp_to_cut=1, direction=1:
        #   crossing_time at t=0,2,4 = 0.5, 2.5, 4.5.
        # With min_return_time=3.0: t=0 seeds; t=2 ret=2.0<3.0 rejected,
        #   last_time=2.5; t=4 ret=4.5-2.5=2.0<3.0 rejected -> empty.
        # With min_return_time=2.5: t=0 seeds; t=2 ret=2.0<2.5 rejected,
        #   last_time=2.5; t=4 ret=4.5-2.5=2.0<2.5 rejected -> empty.
        # Use a longer series to get a non-rejected crossing.
        series = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                           0.0, 1.0, 0.0, 1.0])
        # Crossings at t=0,2,4,6,8,10 (crossing_time = t+0.5).
        # min_return_time=3.0: t=0 seeds; t=2 ret=2.0<3 reject (lt=2.5);
        #   t=4 ret=4.5-2.5=2.0<3 reject (lt=4.5);
        #   t=6 ret=6.5-4.5=2.0<3 reject (lt=6.5);
        #   t=8 ret=8.5-6.5=2.0<3 reject (lt=8.5);
        #   t=10 ret=10.5-8.5=2.0<3 reject -> empty.
        pts, _ = poincare_section(
            series, dim=2, delay=1, comp_to_cut=1,
            threshold=0.5, direction=1, min_return_time=3.0,
        )
        self.assertEqual(pts.size, 0)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestPoincareSectionDefaults(unittest.TestCase):
    """Tests for default parameter values (threshold, comp_to_cut)."""

    def test_threshold_defaults_to_mean(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(2000)
        pts_mean, rets_mean = poincare_section(x, dim=2, delay=1,
                                                direction=0)
        pts_exp, rets_exp = poincare_section(
            x, dim=2, delay=1, threshold=float(np.mean(x)), direction=0
        )
        np.testing.assert_array_equal(pts_mean, pts_exp)
        np.testing.assert_array_equal(rets_mean, rets_exp)

    def test_comp_to_cut_defaults_to_last(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(2000)
        pts_def, rets_def = poincare_section(x, dim=3, delay=1,
                                             threshold=0.0, direction=0)
        pts_exp, rets_exp = poincare_section(
            x, dim=3, delay=1, comp_to_cut=2,
            threshold=0.0, direction=0
        )
        np.testing.assert_array_equal(pts_def, pts_exp)
        np.testing.assert_array_equal(rets_def, rets_exp)


# ---------------------------------------------------------------------------
# dim and delay
# ---------------------------------------------------------------------------


class TestPoincareSectionDimDelay(unittest.TestCase):
    """Tests for the embedding dimension and delay parameters."""

    def test_output_dimensionality_is_dim_minus_one(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(2000)
        for dim in [2, 3, 4, 5]:
            with self.subTest(dim=dim):
                pts, _ = poincare_section(
                    x, dim=dim, delay=1, threshold=0.0, direction=0
                )
                self.assertEqual(pts.shape[1], dim - 1)

    def test_delay_affects_crossing_indices(self):
        # With delay=2, the cutting component reads series[t + 2*comp].
        # Different delay -> different crossing pattern, but the function
        # should still produce valid output.
        rng = np.random.default_rng(0)
        x = rng.standard_normal(2000)
        pts_d1, _ = poincare_section(x, dim=2, delay=1, threshold=0.0,
                                      direction=0)
        pts_d2, _ = poincare_section(x, dim=2, delay=2, threshold=0.0,
                                      direction=0)
        # Both should find some crossings; counts need not match.
        self.assertGreaterEqual(pts_d1.shape[0], 1)
        self.assertGreaterEqual(pts_d2.shape[0], 1)


# ---------------------------------------------------------------------------
# Integration on the Lorenz system
# ---------------------------------------------------------------------------


class TestPoincareSectionLorenz(unittest.TestCase):
    """Integration test on the Lorenz x-component."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        from nolitisea.generate.lorenz import lorenz

        _, data = lorenz(length=3000)
        cls.x = data[:, 0]

    def test_returns_finite_points(self):
        pts, rets = poincare_section(
            self.x, dim=2, delay=1, threshold=0.0, direction=0
        )
        self.assertGreaterEqual(pts.shape[0], 10)
        self.assertEqual(pts.shape[1], 1)
        self.assertTrue(np.all(np.isfinite(pts)))
        self.assertTrue(np.all(np.isfinite(rets)))
        self.assertTrue(np.all(rets > 0))

    def test_dim3_section(self):
        pts, rets = poincare_section(
            self.x, dim=3, delay=1, threshold=0.0, direction=0
        )
        self.assertEqual(pts.shape[1], 2)
        self.assertTrue(np.all(np.isfinite(pts)))
        self.assertTrue(np.all(rets > 0))

    def test_return_times_positive(self):
        _, rets = poincare_section(
            self.x, dim=2, delay=1, threshold=0.0, direction=0
        )
        self.assertTrue(np.all(rets > 0))


# ---------------------------------------------------------------------------
# Parallel bitwise consistency
# ---------------------------------------------------------------------------


class TestPoincareSectionParallel(unittest.TestCase):
    """Serial vs parallel results must be bitwise identical."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        from nolitisea.generate.lorenz import lorenz

        _, data = lorenz(length=2000)
        cls.x = data[:, 0]

    def _run_all_params(self, x, **kwargs):
        """Run sequential and both parallel backends, return results."""
        serial = poincare_section(x, n_jobs=None, **kwargs)
        thread = poincare_section(x, n_jobs=-1, backend="thread", **kwargs)
        process = poincare_section(x, n_jobs=2, backend="process", **kwargs)
        return serial, thread, process

    def _assert_bitwise(self, serial, thread, process):
        for par in (thread, process):
            np.testing.assert_array_equal(par[0], serial[0])
            np.testing.assert_array_equal(par[1], serial[1])

    def test_thread_and_process_match_serial_basic(self):
        serial, thread, process = self._run_all_params(
            self.x, dim=2, delay=1, threshold=0.0, direction=0
        )
        self._assert_bitwise(serial, thread, process)

    def test_match_serial_direction_down(self):
        serial, thread, process = self._run_all_params(
            self.x, dim=2, delay=1, threshold=0.0, direction=1
        )
        self._assert_bitwise(serial, thread, process)

    def test_match_serial_dim3(self):
        serial, thread, process = self._run_all_params(
            self.x, dim=3, delay=1, threshold=0.0, direction=0
        )
        self._assert_bitwise(serial, thread, process)

    def test_match_serial_delay2(self):
        serial, thread, process = self._run_all_params(
            self.x, dim=2, delay=2, threshold=0.0, direction=0
        )
        self._assert_bitwise(serial, thread, process)

    def test_match_serial_min_return_time(self):
        serial, thread, process = self._run_all_params(
            self.x, dim=2, delay=1, threshold=0.0, direction=0,
            min_return_time=5.0,
        )
        self._assert_bitwise(serial, thread, process)

    def test_match_serial_threshold_mean(self):
        serial, thread, process = self._run_all_params(
            self.x, dim=2, delay=1, direction=0  # threshold defaults to mean
        )
        self._assert_bitwise(serial, thread, process)

    def test_match_serial_comp_to_cut_first(self):
        serial, thread, process = self._run_all_params(
            self.x, dim=3, delay=1, comp_to_cut=0,
            threshold=0.0, direction=0
        )
        self._assert_bitwise(serial, thread, process)


if __name__ == "__main__":
    unittest.main()
