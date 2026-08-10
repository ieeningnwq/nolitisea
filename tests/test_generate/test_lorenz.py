import unittest

import numpy as np

from nolitisea.generate.lorenz import lorenz


class TestLorenzSystem(unittest.TestCase):
    def test_default_parameters(self):
        """Test if return types and shapes are correct with default parameters."""
        # Reduce length from 10000 to 100 for faster testing
        t, x = lorenz(length=100)

        self.assertIsInstance(
            t, np.ndarray, "Returned time axis should be a Numpy array"
        )
        self.assertIsInstance(
            x, np.ndarray, "Returned state data should be a Numpy array"
        )
        self.assertEqual(
            len(t), 100, "Length of time array should equal the specified length"
        )
        self.assertEqual(
            x.shape, (100, 3), "Shape of state array should be (length, 3)"
        )

    def test_custom_initial_conditions(self):
        """Test if providing a custom initial state x0 works correctly."""
        custom_x0 = [1.0, 1.0, 1.0]
        _, x = lorenz(length=50, x0=custom_x0)

        self.assertEqual(
            x.shape,
            (50, 3),
            "Return shape should remain correct after passing custom x0",
        )

    def test_zero_discard(self):
        """Test the boundary condition where discard = 0."""
        # If discard=0 and sample equals step (no downsampling),
        # the first point of the returned sequence should be very close to x0
        custom_x0 = [1.0, 1.0, 1.0]
        t, x = lorenz(length=10, x0=custom_x0, step=0.01, sample=0.01, discard=0)

        # Compare the first returned point with the initial value
        np.testing.assert_allclose(
            x[0],
            custom_x0,
            rtol=1e-5,
            err_msg="When discard=0, the sequence start should match the initial state",
        )
        self.assertEqual(t[0], 0.0, "When discard=0, the time axis start should be 0")

    def test_random_seed_reproducibility(self):
        """Test if two calls without x0 generate identical data given a fixed random seed."""
        np.random.seed(42)
        _, x1 = lorenz(length=100)

        np.random.seed(42)
        _, x2 = lorenz(length=100)

        np.testing.assert_array_equal(
            x1, x2, "Generated data must be identical given a fixed random seed"
        )

    def test_custom_parameters(self):
        """Test if the calculation runs normally after modifying lorenz parameters."""
        # Test that modifying parameters does not cause crashes
        try:
            _, _ = lorenz(length=50, sigma=15.0, beta=2.0, rho=10.0)
        except Exception as e:
            self.fail(f"Function raised an exception when modifying parameters: {e}")


if __name__ == "__main__":
    # Run tests
    unittest.main()
