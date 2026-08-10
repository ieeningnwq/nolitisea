# Algorithm reference

This document maps each `nolitisea` module to the corresponding TISEAN
program(s).

| Sub-package | Module | TISEAN program(s) |
|---|---|---|
| `core` | `io` | `get_series`, `get_multi_series`, `search_datafile`, `myfgets` |
| `core` | `neighbors` | `make_box`, `find_neighbors`, `make_multi_box`, `find_multi_neighbors` |
| `core` | `matrix` | `solvele`, `invert_matrix` |
| `core` | `eigen` | `eigen` |
| `core` | `random` | `rand`, `rand_arb_dist` |
| `core` | `rescale` | `rescale_data`, `variance` |
| `core` | `exclude` | `exclude_interval` |
| `core` | `options` | `check_option`, `scan_help`, `test_outfile` |
| `generate` | `henon` | `henon` |
| `generate` | `ikeda` | `ikeda` |
| `generate` | `lorenz` | `lorenz` |
| `generate` | `noise` | `makenoise` |
| `generate` | `ar_run` | `ar-run` |
| `linear` | `ar_model` | `ar-model` |
| `linear` | `arima_model` | `arima-model` |
| `linear` | `autocorr` | `corr` |
| `linear` | `mem_spec` | `mem_spec` |
| `linear` | `spectrum` | `spectrum` |
| `linear` | `pca` | `pca` |
| `linear` | `filters` | `notch`, `wiener`, `low121`, `sav_gol` |
| `utils` | `choose` | `choose` |
| `utils` | `rescale` | `rescale`, `rms` |
| `utils` | `histogram` | `histogram` |
| `utils` | `resample` | `resample` |
| `stationarity` | `recurrence` | `recurr` |
| `stationarity` | `stp` | `stp` |
| `stationarity` | `nstat_z` | `nstat_z` |
| `embedding` | `delay` | `delay` |
| `embedding` | `mutual` | `mutual` |
| `embedding` | `false_nearest` | `false_nearest` |
| `embedding` | `poincare` | `poincare` |
| `embedding` | `extrema` | `extrema` |
| `embedding` | `upo` | `upo`, `upoembed` |
| `prediction` | `local_zeroth` | `lzo-test`, `lzo-run`, `lzo-gm` |
| `prediction` | `local_first` | `lfo-test`, `lfo-run`, `lfo-ar` |
| `prediction` | `rbf` | `rbf` |
| `prediction` | `polynomial` | `polynom`, `polynomp`, `polyback`, `polypar` |
| `noise` | `lazy` | `lazy` |
| `noise` | `ghkss` | `ghkss` |
| `noise` | `compare` | `compare` |
| `dimension` | `d2` | `d2` |
| `dimension` | `c1` | `c1` |
| `dimension` | `boxcount` | `boxcount` |
| `dimension` | `c2_post` | `c2t`, `c2g`, `c2d`, `av-d2` |
| `dimension` | `fsle` | `fsle` |
| `lyapunov` | `lyap_k` | `lyap_k` |
| `lyapunov` | `lyap_r` | `lyap_r` |
| `lyapunov` | `lyap_spec` | `lyap_spec` |
| `surrogates` | `surrogates` | `surrogates` |
| `surrogates` | `randomize` | `randomize` |
| `surrogates` | `endtoend` | `endtoend` |
| `surrogates` | `statistics` | `timerev`, `predict` |
| `spike` | `intervals_events` | `intervals`, `events` |
| `spike` | `spikeauto` | `spikeauto` |
| `spike` | `spikespec` | `spikespec` |
| `spike` | `randomize_spike` | `randomize_spikeauto_exp_random`, `randomize_spikespec_exp_event` |
| `cross` | `xcor` | `xcor` |
| `cross` | `xzero` | `xzero` |
| `cross` | `xc2` | `xc2` |
| `cross` | `xrecur` | `xrecur` |
