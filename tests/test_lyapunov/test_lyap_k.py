"""Tests for nolitisea.lyapunov.lyap_k."""

import unittest

import numpy as np

from nolitisea.generate.henon import henon
from nolitisea.lyapunov.lyap_k import lyap_k


class TestLyapKOutputs(unittest.TestCase):
    def test_output_structure(self):
        x, _ = henon(n=3000)
        res = lyap_k(x, dim=5, delay=1, max_steps=10, n_ref=500,n_jobs=-1)
        self.assertEqual(set(res), {"eps", "dim", "times", "s2", "count"})
        self.assertEqual(res["dim"].tolist(), [2, 3, 4, 5])
        self.assertEqual(res["times"].tolist(), list(range(11)))
        self.assertEqual(res["s2"].shape, (res["eps"].size, 4, 11))
        self.assertEqual(res["count"].shape, (res["eps"].size, 4, 11))
        # Radii ascend and every s2 entry with contributors is finite.
        self.assertTrue(np.all(np.diff(res["eps"]) > 0))
        finite = np.isfinite(res["s2"])
        self.assertTrue(np.all(res["count"][finite] > 0))

    def test_henon_slope_close_to_theory(self):
        # Henon map: the maximal Lyapunov exponent is about 0.4185.
        x, _ = henon(n=20000)
        res = lyap_k(x, dim=5, delay=1, max_steps=20, n_ref=2000,
                     theiler=20,n_jobs=None)
        t = res["times"][1:11]
        slopes = [
            np.polyfit(t, curve[1:11], 1)[0]
            for curve in res["s2"][:, -1, :]
        ]
        # The middle rungs of the eps ladder sit in the scaling region.
        slope = float(np.median(slopes))
        self.assertAlmostEqual(slope, 0.4185, delta=0.05)

    def test_explicit_eps_list_sorted_deduped(self):
        x, _ = henon(n=3000)
        res = lyap_k(x, dim=3, delay=1, max_steps=5, n_ref=300,
                     eps_list=[0.02, 0.01, 0.02])
        self.assertEqual(res["eps"].tolist(), [0.01, 0.02])

    def test_collapsed_ladder_single_radius(self):
        x, _ = henon(n=3000)
        res = lyap_k(x, dim=3, delay=1, max_steps=5, n_ref=300,
                     eps_min=0.5, eps_max=0.1)
        self.assertEqual(res["eps"].tolist(), [0.5])

    def test_invalid_parameters(self):
        x, _ = henon(n=500)
        with self.assertRaises(ValueError):
            lyap_k(x, dim=1, delay=1, max_steps=5)
        with self.assertRaises(ValueError):
            lyap_k(x, dim=5, delay=0, max_steps=5)
        with self.assertRaises(ValueError):
            lyap_k(x, dim=5, delay=1, max_steps=0)
        with self.assertRaises(ValueError):
            lyap_k(x, dim=5, delay=1, max_steps=5, min_dim=6)
        with self.assertRaises(ValueError):
            lyap_k(x, dim=5, delay=1, max_steps=5, theiler=-1)
        with self.assertRaises(ValueError):
            lyap_k(x, dim=5, delay=1, max_steps=5, n_ref=0)
        with self.assertRaises(ValueError):
            lyap_k(x, dim=5, delay=1, max_steps=5, eps_count=0)
        with self.assertRaises(ValueError):
            lyap_k(x, dim=5, delay=1, max_steps=5, eps_list=[])
        with self.assertRaises(ValueError):
            lyap_k(x, dim=5, delay=1, max_steps=5, eps_list=[0.01, -0.1])
        # Series too short for the requested dim/delay/max_steps.
        with self.assertRaises(ValueError):
            lyap_k(np.arange(50.0), dim=5, delay=1, max_steps=46)
        # Constant series.
        with self.assertRaises(RuntimeError):
            lyap_k(np.ones(500), dim=2, delay=1, max_steps=5)
        # Not a 1-D series.
        with self.assertRaises(ValueError):
            lyap_k(np.ones((10, 2)), dim=2, delay=1, max_steps=5)


class TestLyapKParallel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, _ = henon(n=5000)
        cls.args = dict(dim=4, delay=1, max_steps=10, n_ref=1000,
                        theiler=10)
        cls.serial = lyap_k(cls.x, **cls.args)

    def test_thread_backend_bitwise(self):
        par = lyap_k(self.x, n_jobs=-1, **self.args)
        np.testing.assert_array_equal(par["s2"], self.serial["s2"])
        np.testing.assert_array_equal(par["count"], self.serial["count"])
        np.testing.assert_array_equal(par["eps"], self.serial["eps"])

    def test_process_backend_bitwise(self):
        par = lyap_k(self.x, n_jobs=2, backend="process", **self.args)
        np.testing.assert_array_equal(par["s2"], self.serial["s2"])
        np.testing.assert_array_equal(par["count"], self.serial["count"])


if __name__ == "__main__":
    unittest.main()
