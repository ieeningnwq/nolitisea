"""Tests for ``nolitisea.prediction.neural_net``.

Covers MLP, LSTM/GRU, TCN, NARX, Neural ODE, and Transformer.
All torch-dependent tests are skipped when torch is not installed.
"""

import sys
import unittest
from unittest import mock

import numpy as np

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from nolitisea.prediction.neural_net import (
    fit_nn,
    nn_forecast_error,
    predict_nn,
)
from nolitisea.utils.rescale import rescale_data

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sine_series(n=500):
    t = np.linspace(0, 20, n)
    return np.sin(t)


def _linear_map_series(n=300, a=-0.5, b=3.0, x0=10.0):
    x = np.zeros(n)
    x[0] = x0
    for i in range(n - 1):
        x[i + 1] = a * x[i] + b
    return x


def _logistic_map_series(n=500, r=3.7, x0=0.3):
    x = np.zeros(n)
    x[0] = x0
    for i in range(n - 1):
        x[i + 1] = r * x[i] * (1 - x[i])
    return x


def _two_component_sine_cosine(n=500):
    t = np.linspace(0, 20, n)
    return np.column_stack([np.sin(t), np.cos(t)])


# ---------------------------------------------------------------------------
# MLP tests
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestFitNnMlp(unittest.TestCase):
    def test_sine_low_error(self):
        s = _sine_series(500)
        m = fit_nn(s, model_type="mlp", dim=3, hidden_layers=(32, 32),
                    epochs=300, seed=42)
        self.assertLess(m["in_sample_fce"], 0.3)

    def test_returns_expected_keys(self):
        s = _sine_series(200)
        m = fit_nn(s, model_type="mlp", dim=2, hidden_layers=(16,),
                    epochs=50, seed=42)
        for key in ("model_state", "network_config", "model_type",
                     "minv", "interval", "dim", "delay", "step",
                     "n_vars", "in_sample_rmse", "in_sample_fce", "seed"):
            self.assertIn(key, m, f"missing key: {key}")

    def test_model_state_nonempty(self):
        s = _sine_series(200)
        m = fit_nn(s, model_type="mlp", dim=2, hidden_layers=(16,),
                    epochs=50, seed=42)
        self.assertGreater(len(m["model_state"]), 0)
        for v in m["model_state"].values():
            self.assertEqual(v.dtype, np.float64)

    def test_network_config(self):
        s = _sine_series(200)
        m = fit_nn(s, model_type="mlp", dim=3, hidden_layers=(16, 8),
                    epochs=50, seed=42)
        self.assertEqual(m["network_config"]["in_dim"], 3)
        self.assertEqual(m["network_config"]["out_dim"], 1)
        self.assertEqual(m["model_type"], "mlp")


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestFitNnLstm(unittest.TestCase):
    def test_basic_fit(self):
        s = _sine_series(300)
        m = fit_nn(s, model_type="lstm", dim=3, hidden_layers=(32,),
                    epochs=100, seed=42)
        self.assertEqual(m["model_type"], "lstm")
        self.assertTrue(np.isfinite(m["in_sample_fce"]))

    def test_shape(self):
        s = _sine_series(300)
        m = fit_nn(s, model_type="lstm", dim=3, hidden_layers=(32,),
                    epochs=50, seed=42)
        self.assertEqual(m["network_config"]["in_dim"], 3)
        self.assertEqual(m["n_vars"], 1)

    def test_low_error(self):
        s = _sine_series(500)
        m = fit_nn(s, model_type="lstm", dim=3, hidden_layers=(32,),
                    epochs=300, seed=42)
        self.assertLess(m["in_sample_fce"], 0.5)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestFitNnGru(unittest.TestCase):
    def test_basic_fit(self):
        s = _sine_series(300)
        m = fit_nn(s, model_type="gru", dim=3, hidden_layers=(32,),
                    epochs=100, seed=42)
        self.assertEqual(m["model_type"], "gru")
        self.assertTrue(np.isfinite(m["in_sample_fce"]))


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestFitNnTcn(unittest.TestCase):
    def test_basic_fit(self):
        s = _sine_series(300)
        m = fit_nn(s, model_type="tcn", dim=3, hidden_layers=(32,),
                    epochs=100, seed=42)
        self.assertEqual(m["model_type"], "tcn")
        self.assertTrue(np.isfinite(m["in_sample_fce"]))

    def test_shape(self):
        s = _sine_series(300)
        m = fit_nn(s, model_type="tcn", dim=3, hidden_layers=(32,),
                    epochs=50, seed=42)
        self.assertEqual(m["n_vars"], 1)

    def test_low_error(self):
        s = _sine_series(500)
        m = fit_nn(s, model_type="tcn", dim=3, hidden_layers=(32,),
                    epochs=300, seed=42)
        self.assertLess(m["in_sample_fce"], 0.5)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestFitNnNarx(unittest.TestCase):
    def test_basic_fit(self):
        s = _sine_series(300)
        m = fit_nn(s, model_type="narx", dim=3, hidden_layers=(32,),
                    epochs=100, seed=42)
        self.assertEqual(m["model_type"], "narx")
        self.assertTrue(np.isfinite(m["in_sample_fce"]))

    def test_prediction(self):
        s = _sine_series(300)
        m = fit_nn(s, model_type="narx", dim=2, hidden_layers=(16,),
                    epochs=100, seed=42)
        fc = predict_nn(m, s, n_steps=10)
        self.assertEqual(len(fc), 10)
        self.assertTrue(np.all(np.isfinite(fc)))


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestFitNnNeuralOde(unittest.TestCase):
    def test_basic_fit(self):
        s = _sine_series(300)
        m = fit_nn(s, model_type="node", dim=3, hidden_layers=(32,),
                    epochs=100, seed=42)
        self.assertEqual(m["model_type"], "node")
        self.assertTrue(np.isfinite(m["in_sample_fce"]))

    def test_prediction(self):
        s = _sine_series(300)
        m = fit_nn(s, model_type="node", dim=2, hidden_layers=(16,),
                    epochs=100, seed=42)
        fc = predict_nn(m, s, n_steps=10)
        self.assertEqual(len(fc), 10)
        self.assertTrue(np.all(np.isfinite(fc)))


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestFitNnTransformer(unittest.TestCase):
    def test_basic_fit(self):
        s = _sine_series(300)
        m = fit_nn(s, model_type="transformer", dim=3,
                    hidden_layers=(32,), epochs=100, seed=42)
        self.assertEqual(m["model_type"], "transformer")
        self.assertTrue(np.isfinite(m["in_sample_fce"]))

    def test_prediction(self):
        s = _sine_series(300)
        m = fit_nn(s, model_type="transformer", dim=2,
                    hidden_layers=(16,), epochs=100, seed=42)
        fc = predict_nn(m, s, n_steps=10)
        self.assertEqual(len(fc), 10)
        self.assertTrue(np.all(np.isfinite(fc)))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestFitNnReproducibility(unittest.TestCase):
    def test_same_seed_same_weights(self):
        s = _sine_series(200)
        m1 = fit_nn(s, model_type="mlp", dim=2, hidden_layers=(16,),
                    epochs=50, seed=42)
        m2 = fit_nn(s, model_type="mlp", dim=2, hidden_layers=(16,),
                    epochs=50, seed=42)
        for k in m1["model_state"]:
            np.testing.assert_allclose(
                m1["model_state"][k], m2["model_state"][k], atol=1e-10
            )

    def test_different_seed_different_weights(self):
        s = _sine_series(200)
        m1 = fit_nn(s, model_type="mlp", dim=2, hidden_layers=(16,),
                    epochs=50, seed=42)
        m2 = fit_nn(s, model_type="mlp", dim=2, hidden_layers=(16,),
                    epochs=50, seed=43)
        found_diff = False
        for k in m1["model_state"]:
            if np.max(np.abs(m1["model_state"][k]
                              - m2["model_state"][k])) > 1e-10:
                found_diff = True
                break
        self.assertTrue(found_diff)


# ---------------------------------------------------------------------------
# Multivariate
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestFitNnMultivariate(unittest.TestCase):
    def test_two_component_shapes(self):
        s = _two_component_sine_cosine(300)
        m = fit_nn(s, model_type="mlp", dim=2, hidden_layers=(32,),
                    epochs=50, seed=42)
        self.assertEqual(m["n_vars"], 2)
        self.assertEqual(m["network_config"]["in_dim"], 4)
        self.assertEqual(m["network_config"]["out_dim"], 2)

    def test_two_component_low_error(self):
        s = _two_component_sine_cosine(500)
        m = fit_nn(s, model_type="mlp", dim=3, hidden_layers=(64,),
                    epochs=200, seed=42)
        self.assertLess(m["in_sample_fce"], 0.5)


# ---------------------------------------------------------------------------
# In/out sample
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestFitNnInsample(unittest.TestCase):
    def test_out_of_sample_when_insample_lt_n(self):
        s = _sine_series(500)
        m = fit_nn(s, model_type="mlp", dim=3, hidden_layers=(16,),
                    epochs=50, insample=400, seed=42)
        self.assertIsNotNone(m["out_of_sample_rmse"])
        self.assertIsNotNone(m["out_of_sample_fce"])

    def test_out_of_sample_matches_forecast_error(self):
        """Held-out RMSE must pair window ending at t with target t+step."""
        s = _sine_series(500)
        m = fit_nn(s, model_type="mlp", dim=3, hidden_layers=(16,),
                    epochs=50, insample=400, step=1, seed=42)
        s_resc, _, _ = rescale_data(s)
        valid_start = (3 - 1) * 1
        err = nn_forecast_error(s_resc, m, dim=3, delay=1, step=1,
                                i0=400 - 1 - valid_start, i1=500)
        self.assertAlmostEqual(err, m["out_of_sample_rmse"], places=8)

    def test_out_of_sample_matches_forecast_error_step2(self):
        s = _sine_series(500)
        m = fit_nn(s, model_type="mlp", dim=3, hidden_layers=(16,),
                    epochs=50, insample=400, step=2, seed=42)
        s_resc, _, _ = rescale_data(s)
        valid_start = (3 - 1) * 1
        err = nn_forecast_error(s_resc, m, dim=3, delay=1, step=2,
                                i0=400 - 2 - valid_start, i1=500)
        self.assertAlmostEqual(err, m["out_of_sample_rmse"], places=8)

    def test_out_of_sample_none_when_full(self):
        s = _sine_series(200)
        m = fit_nn(s, model_type="mlp", dim=2, hidden_layers=(16,),
                    epochs=50, seed=42)
        self.assertIsNone(m["out_of_sample_rmse"])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestFitNnValidation(unittest.TestCase):
    def test_bad_model_type(self):
        with self.assertRaises(ValueError):
            fit_nn(_sine_series(100), model_type="invalid")

    def test_dim_zero(self):
        with self.assertRaises(ValueError):
            fit_nn(_sine_series(100), dim=0)

    def test_zero_epochs(self):
        with self.assertRaises(ValueError):
            fit_nn(_sine_series(100), epochs=0)

    def test_negative_lr(self):
        with self.assertRaises(ValueError):
            fit_nn(_sine_series(100), lr=-1)

    def test_series_too_short(self):
        with self.assertRaises(ValueError):
            fit_nn(np.array([1.0, 2.0]), dim=3, epochs=10)


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestPredictNn(unittest.TestCase):
    def test_forecast_finite(self):
        s = _sine_series(500)
        m = fit_nn(s, model_type="mlp", dim=3, hidden_layers=(32,),
                    epochs=200, seed=42)
        fc = predict_nn(m, s, n_steps=50)
        self.assertEqual(len(fc), 50)
        self.assertTrue(np.all(np.isfinite(fc)))

    def test_forecast_shape(self):
        s = _sine_series(300)
        m = fit_nn(s, model_type="mlp", dim=2, hidden_layers=(16,),
                    epochs=50, seed=42)
        fc = predict_nn(m, s, n_steps=20)
        self.assertEqual(fc.shape, (20,))

    def test_n_steps_zero(self):
        s = _sine_series(100)
        m = fit_nn(s, model_type="mlp", dim=2, hidden_layers=(16,),
                    epochs=50, seed=42)
        fc = predict_nn(m, s, n_steps=0)
        self.assertEqual(len(fc), 0)

    def test_deterministic(self):
        s = _sine_series(300)
        m = fit_nn(s, model_type="mlp", dim=2, hidden_layers=(16,),
                    epochs=50, seed=42)
        fc1 = predict_nn(m, s, 10)
        fc2 = predict_nn(m, s, 10)
        np.testing.assert_allclose(fc1, fc2)

    def test_forward_pass_shape_all_architectures(self):
        # Regression: sequence models must also return a 1-D (n_vars,)
        # prediction from _forward_pass (no stray batch dimension).
        from nolitisea.prediction.neural_net import (
            _build_from_state,
            _forward_pass,
            _require_torch,
        )

        torch = _require_torch()
        s = _sine_series(200)
        for model_type in ("mlp", "lstm", "gru", "tcn", "narx", "node",
                            "transformer"):
            m = fit_nn(s, model_type=model_type, dim=2, hidden_layers=(16,),
                        epochs=20, seed=42)
            net = _build_from_state(torch, m)
            x_vec = np.zeros(m["n_vars"] * m["dim"])
            pred = _forward_pass(
                torch, net, model_type, x_vec, m["n_vars"], m["dim"],
                m["delay"],
            )
            self.assertEqual(pred.shape, (m["n_vars"],), msg=model_type)

    def test_forecast_shape_multivariate_all_architectures(self):
        s = _two_component_sine_cosine(300)
        for model_type in ("mlp", "lstm", "gru", "tcn", "narx", "node",
                            "transformer"):
            m = fit_nn(s, model_type=model_type, dim=2, hidden_layers=(16,),
                        epochs=20, seed=42)
            fc = predict_nn(m, s, n_steps=10)
            self.assertEqual(fc.shape, (10, 2), msg=model_type)


# ---------------------------------------------------------------------------
# Forecast error
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestNnForecastError(unittest.TestCase):
    def test_matches_fit_nn_insample(self):
        s = _sine_series(300)
        m = fit_nn(s, model_type="mlp", dim=3, hidden_layers=(16,),
                    epochs=50, seed=42)
        s_resc, _, _ = rescale_data(s)
        err = nn_forecast_error(s_resc, m, dim=3, delay=1, step=1,
                                i0=0, i1=300)
        self.assertAlmostEqual(err, m["in_sample_rmse"], places=8)

    def test_empty_interval(self):
        s = _sine_series(100)
        m = fit_nn(s, model_type="mlp", dim=2, hidden_layers=(16,),
                    epochs=50, seed=42)
        s_resc, _, _ = rescale_data(s)
        err = nn_forecast_error(s_resc, m, dim=2, delay=1, step=1,
                                i0=50, i1=50)
        self.assertTrue(np.isnan(err))

    def test_subinterval(self):
        s = _sine_series(300)
        m = fit_nn(s, model_type="mlp", dim=3, hidden_layers=(16,),
                    epochs=50, seed=42)
        s_resc, _, _ = rescale_data(s)
        err = nn_forecast_error(s_resc, m, dim=3, delay=1, step=1,
                                i0=100, i1=200)
        self.assertTrue(np.isfinite(err))
        self.assertGreater(err, 0)


# ---------------------------------------------------------------------------
# Brute-force agreement
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestBruteForceAgreement(unittest.TestCase):
    def test_linear_map_recovery(self):
        s = _linear_map_series(300)
        m = fit_nn(s, model_type="mlp", dim=2, hidden_layers=(16,),
                    epochs=400, seed=42)
        self.assertLess(m["in_sample_rmse"], 0.5)

    def test_logistic_map_one_step(self):
        s = _logistic_map_series(400)
        m = fit_nn(s, model_type="mlp", dim=2, hidden_layers=(64, 32),
                    epochs=400, seed=42)
        fc = predict_nn(m, s, n_steps=1)
        expected = 3.7 * s[-1] * (1 - s[-1])
        self.assertLess(abs(fc[0] - expected), 0.1)


# ---------------------------------------------------------------------------
# Lazy import
# ---------------------------------------------------------------------------

class TestLazyImport(unittest.TestCase):
    def test_module_imports_without_torch(self):
        """Module should be importable without torch installed."""
        import nolitisea.prediction.neural_net as nn_mod
        self.assertTrue(hasattr(nn_mod, "fit_nn"))
        self.assertTrue(hasattr(nn_mod, "predict_nn"))
        self.assertTrue(hasattr(nn_mod, "nn_forecast_error"))

    @unittest.skipUnless(_HAS_TORCH, "torch not installed")
    def test_fit_nn_raises_without_torch(self):
        """fit_nn should raise ImportError when torch is unavailable."""
        import nolitisea.prediction.neural_net as nn_mod
        original_torch = nn_mod._TORCH
        nn_mod._TORCH = None
        try:
            with mock.patch.dict(sys.modules, {"torch": None}):
                with self.assertRaises(ImportError) as ctx:
                    fit_nn(_sine_series(100), epochs=5)
                self.assertIn("nolitisea[nn]", str(ctx.exception))
        finally:
            nn_mod._TORCH = original_torch


if __name__ == "__main__":
    unittest.main()
