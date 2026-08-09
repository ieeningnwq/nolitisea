import unittest
import numpy as np

from nolitisea.delay import mutual_information


class TestMutualInformation(unittest.TestCase):

    def test_independent_variables(self):
        """Mutual information of independent variables should be ~0."""
        np.random.seed(42)
        x = np.random.randn(5000)
        y = np.random.randn(5000)

        mi = mutual_information(x, y, bins=20, normalize=False)
        self.assertLess(abs(mi), 0.05)

        nmi = mutual_information(x, y, bins=20, normalize=True)
        self.assertLess(nmi, 0.05)

    def test_identical_variables(self):
        """MI(X, X) should equal entropy H(X)."""
        np.random.seed(42)
        x = np.random.rand(1000)

        mi = mutual_information(x, x, bins=20, normalize=False)

        # Rough check: MI(X,X) > 0
        self.assertGreater(mi, 0.5)

        nmi = mutual_information(x, x, bins=20, normalize=True)
        self.assertAlmostEqual(nmi, 1.0, places=2)

    def test_linear_transform_invariance(self):
        """MI(X, aX + b) = MI(X, X)."""
        np.random.seed(42)
        x = np.random.randn(2000)
        y = 3.5 * x + 7.2

        mi1 = mutual_information(x, x, bins=20)
        mi2 = mutual_information(x, y, bins=20)

        self.assertAlmostEqual(mi1, mi2, places=2)

    def test_normalize_range(self):
        """Normalized MI must lie in [0, 1]."""
        np.random.seed(42)
        x = np.random.randn(1000)
        y = np.random.randn(1000)

        nmi = mutual_information(x, y, bins=15, normalize=True)
        self.assertGreaterEqual(nmi, 0.0)
        self.assertLessEqual(nmi, 1.0)

    def test_constant_signal(self):
        """MI with a constant signal should be 0."""
        x = np.random.randn(500)
        y = np.ones(500)

        mi = mutual_information(x, y, bins=10)
        self.assertAlmostEqual(mi, 0.0, places=6)

        nmi = mutual_information(x, y, bins=10, normalize=True)
        self.assertAlmostEqual(nmi, 0.0, places=6)

    def test_empty_input(self):
        """Empty arrays should return MI = 0 without crashing."""
        x = np.array([])
        y = np.array([])

        mi = mutual_information(x, y)
        self.assertEqual(mi, 0.0)

    def test_length_mismatch_raises(self):
        """Mismatched lengths should raise ValueError."""
        x = np.random.randn(100)
        y = np.random.randn(90)

        with self.assertRaises(ValueError):
            mutual_information(x, y)

    def test_bins_stability(self):
        """Changing bins should not produce NaNs or Infs."""
        np.random.seed(42)
        x = np.random.randn(2000)
        y = x + 0.1 * np.random.randn(2000)

        for b in [5, 10, 20, 50]:
            with self.subTest(bins=b):
                mi = mutual_information(x, y, bins=b)
                self.assertTrue(np.isfinite(mi))

    def test_log_base_two(self):
        """MI should be in bits (base-2)."""
        # Binary variable: perfect correlation
        x = np.array([0, 1, 0, 1])
        y = np.array([0, 1, 0, 1])

        mi = mutual_information(x, y, bins=2, normalize=False)
        # Max MI = 1 bit for binary variables
        self.assertAlmostEqual(mi, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
