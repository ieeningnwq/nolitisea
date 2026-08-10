# nolitisea

A Python toolkit for nonlinear time series analysis, reimplementing the
functionality of the TISEAN package in Python.

## Installation

```bash
pip install -e .
```

## Usage

```python
import numpy as np
from nolitisea.core.series import TimeSeries
```

## License

GPL-2.0-or-later (compatible with original TISEAN license).


我已经在前面的对话中完整阅读了 Tisean 的源码结构（`source_c/`、`source_c/routines/`、`source_f/`）和文档目录（`docs/contents.html`、`docs/alphabetical.html`）。下面仅做结构设计，不创建任何文件或文件夹。

---

## 一、Tisean 架构要点（设计依据）

1. **双层组织**：主算法（`source_c/*.c` + `source_f/*.f`）+ 共享例程（`source_c/routines/`）。
2. **共享例程是基石**：[tsa.h](file:///Users/wenqingnie/Downloads/Tisean_3.0.1/source_c/routines/tsa.h) 暴露的 `get_series`、`find_neighbors`、`make_box`、`solvele`、`invert_matrix`、`eigen`、`rescale_data`、`variance`、`exclude_interval`、`rand` 等被几乎所有算法复用。
3. **功能分区**（来自 [contents.html](file:///Users/wenqingnie/Downloads/Tisean_3.0.1/docs/contents.html)）：数据生成 / 线性工具 / 实用工具 / 平稳性 / 嵌入 / 预测 / 降噪 / 维数与熵 / Lyapunov / 替代数据 / 脉冲序列 / 交叉分析。
4. **命令行工具集**：每个算法是独立程序，统一参数（`-l -x -c -m -d -o -V`）。

---

## 二、`nolitisea` 项目结构设计

采用 `src` layout + `pyproject.toml`，按 Tisean 功能分类划分子包，核心抽取到 `core`，提供库 API + CLI 双入口。

```
nolitisea/
├── pyproject.toml
├── README.md
├── LICENSE
├── docs/
│   └── algorithms.md               # 算法说明与 Tisean 对照表
├── tests/
│   ├── conftest.py
│   ├── test_core.py
│   ├── test_embedding.py
│   ├── test_lyapunov.py
│   └── data/
└── src/
    └── nolitisea/
        ├── __init__.py
        ├── __main__.py
        ├── cli.py
        │
        ├── core/                   # 共享基础设施（对应 source_c/routines/）
        │   ├── __init__.py
        │   ├── io.py
        │   ├── series.py
        │   ├── embed.py
        │   ├── neighbors.py
        │   ├── boxcount_kernel.py
        │   ├── matrix.py
        │   ├── eigen.py
        │   ├── random.py
        │   ├── rescale.py
        │   ├── exclude.py
        │   └── options.py
        │
        ├── generate/               # 数据生成（henon/ikeda/lorenz/makenoise）
        │   ├── __init__.py
        │   ├── henon.py
        │   ├── ikeda.py
        │   ├── lorenz.py
        │   ├── noise.py
        │   └── ar_run.py
        │
        ├── linear/                 # 线性工具
        │   ├── __init__.py
        │   ├── ar_model.py
        │   ├── arima_model.py
        │   ├── autocorr.py
        │   ├── mem_spec.py
        │   ├── spectrum.py
        │   ├── pca.py
        │   └── filters.py
        │
        ├── utils/                  # 数据预处理
        │   ├── __init__.py
        │   ├── choose.py
        │   ├── rescale.py
        │   ├── histogram.py
        │   └── resample.py
        │
        ├── stationarity/           # 平稳性分析
        │   ├── __init__.py
        │   ├── recurrence.py
        │   ├── stp.py
        │   └── nstat_z.py
        │
        ├── embedding/              # 嵌入与 Poincaré
        │   ├── __init__.py
        │   ├── delay.py
        │   ├── mutual.py
        │   ├── false_nearest.py
        │   ├── poincare.py
        │   ├── extrema.py
        │   └── upo.py
        │
        ├── prediction/             # 相空间预测
        │   ├── __init__.py
        │   ├── local_zeroth.py
        │   ├── local_first.py
        │   ├── rbf.py
        │   └── polynomial.py
        │
        ├── noise/                  # 非线性降噪
        │   ├── __init__.py
        │   ├── lazy.py
        │   ├── ghkss.py
        │   └── compare.py
        │
        ├── dimension/              # 维数与熵
        │   ├── __init__.py
        │   ├── d2.py
        │   ├── c1.py
        │   ├── boxcount.py
        │   ├── c2_post.py
        │   └── fsle.py
        │
        ├── lyapunov/               # Lyapunov 指数
        │   ├── __init__.py
        │   ├── lyap_k.py
        │   ├── lyap_r.py
        │   └── lyap_spec.py
        │
        ├── surrogates/             # 替代数据
        │   ├── __init__.py
        │   ├── surrogates.py
        │   ├── randomize.py
        │   ├── endtoend.py
        │   └── statistics.py
        │
        ├── spike/                  # 脉冲序列分析
        │   ├── __init__.py
        │   ├── intervals_events.py
        │   ├── spikeauto.py
        │   ├── spikespec.py
        │   └── randomize_spike.py
        │
        └── cross/                  # 交叉分析（XTisean）
            ├── __init__.py
            ├── xcor.py
            ├── xzero.py
            ├── xc2.py
            └── xrecur.py
```

---

## 三、各文件函数设计（签名 + 作用）

### 顶层

| 文件 | 函数 | 作用 |
|---|---|---|
| `__init__.py` | — | 导出 `TimeSeries`、版本号 |
| `__main__.py` | — | 支持 `python -m nolitisea` |
| `cli.py` | `build_parser()` | 构建 argparse 子命令 |
| | `main(argv)` | 分发到子模块 `run()` |
| | `_add_common(parser)` | 注册公共参数 `-l/-x/-c/-o/-V` |

### core/（对应 `source_c/routines/`）

| 文件 | 函数 | 作用 | Tisean 对应 |
|---|---|---|---|
| `io.py` | `read_series(path, column, length, exclude)` | 读单列序列 | `get_series` |
| | `read_multi_series(path, columns, length, exclude)` | 读多列 | `get_multi_series` |
| | `write_series(path, data, header)` | 写文件 | — |
| | `search_datafile(argv)` | 定位输入文件名 | `search_datafile` |
| | `myfgets(stream, max_len)` | 安全读行 | `myfgets` |
| `series.py` | `class TimeSeries` | 单/多变量数据容器 | — |
| `embed.py` | `delay_embedding(series, dim, delay)` | 延迟嵌入矩阵 | delay 逻辑 |
| | `mixed_embedding(series_list, dims, delays)` | 多变量混合嵌入 | — |
| | `embedding_indices(n, dim, delay)` | 合法嵌入索引 | — |
| `neighbors.py` | `make_box(series, dim, delay, eps, box_size)` | 盒哈希索引 | `make_box` |
| | `find_neighbors(series, point, eps, dim, delay, box)` | eps 邻域查询 | `find_neighbors` |
| | `find_neighbors_kdtree(points, query, eps)` | KDTree 备选实现 | — |
| | `find_multi_neighbors(...)` | 多变量邻居 | `find_multi_neighbors` |
| | `make_multi_box(...)` | 多变量盒索引 | `make_multi_box` |
| `boxcount_kernel.py` | `assign_boxes(points, eps)` | 点→盒子坐标 | — |
| | `box_neighbor_offsets(dim)` | 3^dim 邻接偏移 | — |
| `matrix.py` | `solve_linear(a, b)` | 解 A x = b | `solvele` |
| | `invert_matrix(a)` | 方阵求逆 | `invert_matrix` |
| | `pseudo_inverse(a)` | 伪逆（过定拟合） | — |
| `eigen.py` | `eig_sym(a)` | 对称矩阵特征系 | `eigen` |
| | `jacobi_eigen(a, max_iter)` | 雅可比法（教学） | — |
| `random.py` | `rnd_init(seed)` | 初始化种子 | `rnd_init` |
| | `gaussian(size)` | 正态随机数 | `gaussian` |
| | `uniform(size)` | 均匀随机数 | — |
| | `rnd_long()` | 长整型伪随机 | `rnd_long` |
| | `rand_arb_dist(values, weights, size)` | 任意分布采样 | `rand_arb_dist` |
| `rescale.py` | `rescale_data(series, lo, hi)` | 线性缩放 | `rescale_data` |
| | `variance(series)` | 均值与方差 | `variance` |
| | `rms_normalize(series)` | 去均值除以 std | — |
| `exclude.py` | `exclude_interval(length, start, end)` | 排除区间索引 | `exclude_interval` |
| `options.py` | `check_option(argv, name, n_args)` | 解析选项 | `check_option` |
| | `scan_help(argv)` | 检测 -h | `scan_help` |
| | `test_outfile(path)` | 测试可写 | `test_outfile` |

### generate/（数据生成）

| 文件 | 函数 | Tisean 对应 |
|---|---|---|
| `henon.py` | `henon(n, a, b, x0, y0)` / `run(argv)` | henon |
| `ikeda.py` | `ikeda(n, u, x0, y0)` / `run(argv)` | ikeda |
| `lorenz.py` | `lorenz(n, dt, sigma, rho, beta, x0)` / `run(argv)` | lorenz |
| `noise.py` | `add_noise(series, percent, seed)` / `add_uniform_noise(...)` / `run(argv)` | makenoise |
| `ar_run.py` | `ar_run(coeffs, n, x_init, noise_std)` / `run(argv)` | ar-run |

### linear/（线性工具）

| 文件 | 函数 | Tisean 对应 |
|---|---|---|
| `ar_model.py` | `fit_ar(series, order)` / `iterate_ar(...)` / `run` | ar-model |
| `arima_model.py` | `fit_arima(series, p, d, q)` / `difference(...)` / `run` | arima-model |
| `autocorr.py` | `autocorrelation(series, max_lag)` / `run` | corr |
| `mem_spec.py` | `mem_spectrum(series, order, n_freq)` / `run` | mem_spec |
| `spectrum.py` | `power_spectrum(series, n_per_seg, window)` / `run` | spectrum |
| `pca.py` | `pca(data, n_components)` / `svd_truncate(...)` / `run` | pca |
| `filters.py` | `notch_filter(...)` / `wiener_filter(...)` / `low121(...)` / `savitzky_golay(...)` / `run` | notch / wiener / low121 / sav_gol |

### utils/（实用工具）

| 文件 | 函数 | Tisean 对应 |
|---|---|---|
| `choose.py` | `choose_rows(...)` / `choose_columns(...)` / `run` | choose |
| `rescale.py` | `rescale(...)` / `rms(...)` / `run` | rescale / rms |
| `histogram.py` | `histogram(series, bins, rng)` / `run` | histogram |
| `resample.py` | `resample(series, factor)` / `resample_interp(...)` / `run` | resample |

### stationarity/（平稳性）

| 文件 | 函数 | Tisean 对应 |
|---|---|---|
| `recurrence.py` | `recurrence_matrix(...)` / `recurrence_rate(...)` / `run` | recurr |
| `stp.py` | `space_time_separation(...)` / `run` | stp |
| `nstat_z.py` | `nstat_z(series, dim, delay, n_parts)` / `run` | nstat_z |

### embedding/（嵌入与 Poincaré）

| 文件 | 函数 | Tisean 对应 |
|---|---|---|
| `delay.py` | `delay_vectors(...)` / `run` | delay |
| `mutual.py` | `mutual_information(series, max_lag, n_bins)` / `first_minimum(...)` / `run` | mutual |
| `false_nearest.py` | `false_nearest(series, dim_max, delay, rt, fs)` / `run` | false_nearest |
| `poincare.py` | `poincare_section(series, axis, condition)` / `run` | poincare |
| `extrema.py` | `extrema(series, mode)` / `run` | extrema |
| `upo.py` | `find_upo(...)` / `upo_embed(...)` / `run` | upo / upoembed |

### prediction/（预测）

| 文件 | 函数 | Tisean 对应 |
|---|---|---|
| `local_zeroth.py` | `lzo_test(...)` / `lzo_run(...)` / `lzo_gm(...)` / `run` | lzo-test / lzo-run / lzo-gm |
| `local_first.py` | `lfo_test(...)` / `lfo_run(...)` / `lfo_ar(...)` / `run` | lfo-test / lfo-run / lfo-ar |
| `rbf.py` | `fit_rbf(...)` / `predict_rbf(...)` / `run` | rbf |
| `polynomial.py` | `fit_polynom(...)` / `polyback(...)` / `polypar(...)` / `run` | polynom / polyback / polypar / polynomp |

### noise/（非线性降噪）

| 文件 | 函数 | Tisean 对应 |
|---|---|---|
| `lazy.py` | `lazy_noise_reduction(...)` / `run` | lazy |
| `ghkss.py` | `ghkss(...)` / `run` | ghkss |
| `compare.py` | `compare(a, b)` / `run` | compare |

### dimension/（维数与熵）

| 文件 | 函数 | Tisean 对应 |
|---|---|---|
| `d2.py` | `correlation_integral(...)` / `d2(...)` / `run` | d2 |
| `c1.py` | `fixed_mass_d1(...)` / `run` | c1 |
| `boxcount.py` | `renyi_entropy(...)` / `run` | boxcount |
| `c2_post.py` | `c2t(...)` / `c2g(...)` / `c2d(...)` / `av_d2(...)` | c2t / c2g / c2d / av-d2 |
| `fsle.py` | `fsle(...)` / `run` | fsle |

### lyapunov/（Lyapunov 指数）

| 文件 | 函数 | Tisean 对应 |
|---|---|---|
| `lyap_k.py` | `lyap_k(...)` / `run` | lyap_k |
| `lyap_r.py` | `lyap_r(...)` / `run` | lyap_r |
| `lyap_spec.py` | `lyap_spec(...)` / `run` | lyap_spec |

### surrogates/（替代数据）

| 文件 | 函数 | Tisean 对应 |
|---|---|---|
| `surrogates.py` | `aaft(...)` / `iterative_aaft(...)` / `run` | surrogates |
| `randomize.py` | `randomize(series, cost_func, n_iter)` / `run` | randomize |
| `endtoend.py` | `endtoend(series, dim, delay)` / `run` | endtoend |
| `statistics.py` | `time_reversibility(series)` / `predict_stat(...)` | timerev / predict |

### spike/（脉冲序列）

| 文件 | 函数 | Tisean 对应 |
|---|---|---|
| `intervals_events.py` | `events_to_intervals(...)` / `intervals_to_events(...)` / `run` | intervals / events |
| `spikeauto.py` | `spike_autocorrelation(...)` / `run` | spikeauto |
| `spikespec.py` | `spike_spectrum(...)` / `run` | spikespec |
| `randomize_spike.py` | `randomize_spikeauto(...)` / `randomize_spikespec(...)` / `run` | randomize_spikeauto_exp_random / randomize_spikespec_exp_event |

### cross/（交叉分析）

| 文件 | 函数 | Tisean 对应 |
|---|---|---|
| `xcor.py` | `cross_correlation(a, b, max_lag)` / `run` | xcor |
| `xzero.py` | `xzero(a, b, dim, delay, eps)` / `run` | xzero |
| `xc2.py` | `cross_correlation_integral(...)` / `run` | xc2 |
| `xrecur.py` | `cross_recurrence(...)` / `run` | xrecur |

---

## 四、设计原则

1. **核心复用**：`core/` 对应 `routines/`，所有算法共享 `delay_embedding` 与 `find_neighbors`，与 Tisean 一致。
2. **库 API + CLI 双形态**：每个模块除核心函数外提供 `run(argv=None)`，`cli.py` 注册为子命令；函数返回 NumPy 数组而非写文件，便于 Jupyter 分析。
3. **参数命名沿用 Tisean**：`-m/-d/-l/-x/-c/-o/-V`，降低迁移成本。
4. **依赖**：`numpy`（必需）、`scipy`（FFT/KDTree/优化，替代 SLATEC）、`numba`（JIT 加速盒搜索）、`matplotlib`（可视化，可选）。
5. **分阶段实现优先级**：① `core/` + `generate/` + `embedding/`（基础）→ ② `lyapunov/` + `dimension/` + `stationarity/`（经典非线性量，复用 neighbors）→ ③ `prediction/` + `noise/` + `surrogates/` → ④ `linear/` + `utils/` + `spike/` + `cross/`。

---

结构设计完成，尚未创建任何文件或文件夹。确认无误后告诉我，我再开始创建文件并填充实现。