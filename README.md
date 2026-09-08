# nolitisea

A Python toolkit for **nonlinear time series analysis**.

`nolitisea` bundles the classical work flow of nonlinear dynamics into a single,
dependency-light library: reconstruct a scalar measurement in phase space,
choose embedding parameters, estimate dimensions and Lyapunov exponents,
forecast with local or neural models, test for stationarity and nonlinearity,
reduce noise, and generate benchmark dynamical systems.  Multivariate series
are supported throughout, and computationally heavy routines ship with
built-in parallelism.

## Features

- **Phase-space reconstruction** — delay embedding, mixed/multivariate embedding, Theiler-window-aware neighbour search
- **Embedding parameter selection** — mutual information and kernel/Renyi MI for the time delay, Kennel and Cao false-nearest-neighbour methods for the embedding dimension, the C-C method, Poincare sections
- **Invariant estimates** — correlation sums and correlation dimension (`c1`, `c2`, `d2`), largest Lyapunov exponent (`lyap_r`, `lyap_k`), full Lyapunov spectrum (`lyap_spec`), finite-size Lyapunov exponents (`fsle`)
- **Entropy** — sample entropy, Renyi/box-counting entropy, transfer entropy and multivariate transfer entropy
- **Prediction** — zeroth- and first-order local predictors (`lzo_*`, `lfo_*`), RBF networks, polynomial models, echo state networks, and PyTorch neural networks (MLP, LSTM, GRU, TCN, NARX, neural ODE, transformer) behind a unified `fit_*` / `predict_*` API
- **Stationarity** — cross-prediction nonstationarity test (`nstat_z`), space-time separation plot (`stp`) for Theiler-window selection, recurrence plots and recurrence quantification analysis (RR, DET, LAM, TT, ENTR, ...)
- **Surrogates & nonlinearity tests** — Fourier, AAFT and IAAFT surrogates, end-to-end nonlinearity testing, time-reversibility and prediction statistics
- **Cross-analysis** — cross-zeroth prediction, cross-recurrence, cross-correlation and cross-correlation integrals for coupled/driver-response series
- **Noise reduction & linear tools** — GHKSS and local-average denoising, synthetic noise, error metrics, AR/ARIMA models, power and maximum-entropy spectra, autocorrelation, notch/Wiener/Savitzky-Golay filters
- **Benchmark systems** — Lorenz, Rossler, Henon, Ikeda, logistic map and Mackey-Glass generators
- **Performance** — `cKDTree`-based neighbour searches and an optional `n_jobs` / `backend` ("thread" or "process") parallel layer for the expensive scans

## Installation

Install from PyPI:

```bash
pip install nolitisea
```

Alternatively, install from a local clone:

```bash
git clone https://github.com/ieeningnwq/nolitisea.git
cd nolitisea
pip install .
```

For development or running the example notebooks, an editable install is
recommended:

```bash
pip install -e .
```

Core dependencies (NumPy, SciPy) are installed automatically.  The
neural-network predictors additionally need PyTorch, which is an optional
extra (imported lazily, only needed when you call the `nolitisea.prediction`
neural-network functions):

```bash
pip install "nolitisea[nn]"
```

Requires Python 3.10 or newer.

## Quickstart

```python
import numpy as np
from nolitisea.generate.lorenz import lorenz

# Generate a benchmark trajectory (sampled Lorenz system).
t, states = lorenz(length=2000)
x = states[:, 0]
```

Choose embedding parameters — the first minimum of the time-delayed mutual
information gives the delay, and the false-nearest-neighbour fraction gives
the embedding dimension:

```python
from nolitisea.embedding.mutual import first_minimum
from nolitisea.embedding.false_nearest import kennel_method

tau = first_minimum(x, max_lag=100)
fnn = kennel_method(x, min_emb=1, max_emb=8, delay=tau)
print("delay:", tau)
```

Estimate the largest Lyapunov exponent (the average logarithmic divergence
of nearby trajectories should grow linearly for chaotic data):

```python
from nolitisea.lyapunov.lyap_r import lyap_r

lam = lyap_r(x, dim=5, delay=tau, max_steps=20, theiler=tau, n_jobs=-1)
# lam["times"], lam["divergence"]
```

Forecast the series with a radial-basis-function model:

```python
from nolitisea.prediction.rbf import fit_rbf, predict_rbf

model = fit_rbf(x, dim=5, delay=tau, n_centers=50, insample=1800)
forecast = predict_rbf(model, x[:1800], n_steps=50)
```

Or with a neural network (requires the `[nn]` extra):

```python
from nolitisea.prediction.neural_net import fit_nn, predict_nn

nn = fit_nn(x[:1500], model_type="mlp", dim=3, delay=tau,
            hidden_layers=(32,), epochs=200, seed=0)
nn_forecast = predict_nn(nn, x[:1500], n_steps=20)
```

Test stationarity with the cross-prediction error matrix — stationary data
give a roughly uniform matrix, while nonstationary data show a diagonal
valley:

```python
from nolitisea.stationarity.nstat_z import nstat_z

rng = np.random.default_rng(0)
result = nstat_z(rng.standard_normal(2000), dim=3, delay=1,
                 n_pieces=4, min_neighbors=20)
print(result["matrix"].shape)  # (4, 4) normalised forecast-error matrix
```

Test for nonlinearity against surrogates:

```python
from nolitisea.surrogates.surrogates import aaft

surrogate = aaft(x[:1000])  # amplitude-adjusted Fourier surrogate
```

`iaaft` (iterative refinement) and the `endtoend` nonlinearity test are also
available in `nolitisea.surrogates`.

## Module reference

| Module | Highlights |
|---|---|
| `nolitisea.generate` | `lorenz`, `roessler`, `henon`, `ikeda`, `logistic`, `mackey_glass` |
| `nolitisea.core` | `delay_embedding`, `lag_block_delay_embed`, `delay_vectors`, `mixed_embedding`, `find_neighbors` |
| `nolitisea.embedding` | `first_minimum`, `embedded_mutual_information`, `matrix_renyi_mutual_information`, `cc_method`, `kennel_method`, `cao_method`, `poincare_section` |
| `nolitisea.dimension` | `c1`, `correlation_integral`, `d2` |
| `nolitisea.lyapunov` | `lyap_r`, `lyap_k`, `lyap_spec`, `fsle` |
| `nolitisea.entropy` | `sample_entropy`, `renyi_entropy`, `transfer_entropy`, `multivariate_transfer_entropy` |
| `nolitisea.prediction` | `lzo_run`/`lzo_test`/`lzo_gm`, `lfo_run`/`lfo_test`/`lfo_ar`, `fit_rbf`/`predict_rbf`, `fit_polynom`, `fit_esn`/`predict_esn`, `fit_nn`/`predict_nn` |
| `nolitisea.stationarity` | `nstat_z`, `stp`, `recurr`, `recurrence_matrix` plus RQA metrics (`recurrence_rate`, `determinism`, `laminarity`, `trapping_time`, ...) |
| `nolitisea.surrogates` | `ft`, `aaft`, `iaaft`, `endtoend`, `time_reversibility`, `predict_stat` |
| `nolitisea.cross` | `cross_zeroth`, `cross_recurrence`, `cross_correlation`, `cross_correlation_integral` |
| `nolitisea.linear` | `fit_ar_model`, `fit_arima`, `autocorrelation`, `power_spectrum`, `mem_spectrum`, `notch_filter`, `wiener_filter`, `savitzky_golay` |
| `nolitisea.noise` | `ghkss`, `lazy`, `add_noise`, error metrics (`mae`, `mse`, `rmse`, `r2_score`, ...) |
| `nolitisea.utils` | `rescale_data`, `resample`, `parallel_map`, pairwise distance helpers |

Every public function has a numpydoc docstring describing parameters,
returns, conventions (Chebyshev metric, per-component `[0, 1]` rescaling,
Theiler windows) and references.

## Examples

Runnable, end-to-end Jupyter notebooks live in
[`examples/`](examples/):

- `generate.ipynb` — benchmark dynamical systems
- `embedding.ipynb` — delay, dimension and Poincare analysis
- `dimension.ipynb` — correlation sums and dimension estimates
- `lyapunov.ipynb` — Lyapunov exponents and spectra
- `entropy.ipynb` — entropies and information transfer
- `prediction.ipynb` — local, RBF, polynomial, ESN and neural predictors
- `stationarity.ipynb` — recurrence plots, space-time separation, cross-prediction test
- `surrogates.ipynb` — surrogate generation and nonlinearity tests
- `cross.ipynb`, `linear.ipynb`, `noise.ipynb`, `core.ipynb`, `utils.ipynb`

## Testing

The test suite is written with `unittest` and cross-checks the vectorised
implementations against independent brute-force references:

```bash
python -m unittest discover -s tests
```

## References

The implemented methods follow the standard literature on nonlinear time
series analysis:

- Kantz, H., & Schreiber, T. (2004). *Nonlinear Time Series Analysis*. Cambridge University Press.
- Hegger, R., Kantz, H., & Schreiber, T. (1999). Practical implementation of nonlinear time series methods. *Chaos*, 9(2), 413–435.
- Kennel, M. B., Brown, R., & Abarbanel, H. D. I. (1992). Determining embedding dimension for phase-space reconstruction using a geometrical construction. *Phys. Rev. A*, 45, 3403.
- Cao, L. (1997). Practical method for determining the minimum embedding dimension of a scalar time series. *Physica D*, 110, 43–50.
- Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993). A practical method for calculating largest Lyapunov exponents. *Physica D*, 65, 117–134.
- Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. *Physics Reports*, 438, 237–329.

## License

See [LICENSE](LICENSE).
