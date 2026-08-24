# Sequential Model Averaging for Epidemic Modelling (BMA-SMC²)

Code accompanying the paper on a sequential ensemble methodology for epidemic
modelling that combines discrete-time Hawkes processes (DTHP) and SEIR
models, using Sequential Monte Carlo Squared (SMC²) with a sliding-window
model-averaging (MA) scheme. Evaluated on simulated scenarios and on the
2024 Irish influenza and Irish COVID-19 epidemics.

## Repository structure

```
src/                    core algorithm — model definitions, particle filter,
                         SMC², PMMH, priors, observation models, plotting,
                         and the shared "run + plot" logic
simulation_study/        one .py (config: priors, data-generating process) +
                         one .ipynb (interactive run + plots) per scenario (A, B, C)
real_data_study/         one .py (config: priors, data loading) + one .ipynb
                         (interactive run + plots) per real-data application (flu, COVID)
data/
  simulated/scenario_*/   cached CSV replicates (auto-generated, see below)
  real/                   Influenza2024.xlsx and the COVID CSV (included)
figures/
  simulated/scenario_*/   figures produced by the simulation_study notebooks
  real/flu/, real/covid/  figures produced by the real_data_study notebooks
results/                  cached BMA-SMC² output (.pkl), see "Restyling plots" below
```

### `src/`

| file                  | contents |
|------------------------|----------|
| `epidemic_models.py`   | stochastic SEIR, SEIRS, and DTHP transition models |
| `state_process.py`     | generic vectorised forward-propagation for SEIR-family models |
| `resampling.py`        | particle resampling schemes (stratified, systematic, residual, multinomial) |
| `priors.py`            | prior draws, parameter transforms (log/logit), particle initialisation |
| `observation_models.py`| observation likelihoods (Poisson, normal, NB, normal-approx-NB) |
| `particle_filter.py`   | single-model particle filter (used inside PMMH and for final forecasts) |
| `pmmh.py`               | PMMH rejuvenation kernel + log-prior |
| `bma_smc2.py`           | `BMA_SMC2`: the main two-model SMC² sampler |
| `helpers.py`            | `compute_window_weights` (model weights over time), `compute_model_average`, `extend_array` |
| `visualization.py`      | `trace_smc`, `plot_smc`, `corrected_matrix` |
| `simulate.py`           | data-generating processes + replicate CSV cache (generate/load) |
| `scenario_runner.py`    | shared "run BMA_SMC2 on one dataset, then plot" logic used by every scenario/application script |

## Running

Each scenario/application is a **notebook** (`.ipynb`) that imports its
config from the matching `.py` file in the same folder (priors, the
data-generating process or data loader, particle counts, file paths).
Open Jupyter, `cd` into `simulation_study/` or `real_data_study/`, and open
the notebook you want:

```bash
pip install -r requirements.txt
cd simulation_study        # or real_data_study
jupyter lab                # or jupyter notebook
```

Every notebook follows the same six-cell layout:

1. **Setup** — points Python at the repo root so `src.` imports and the
   `data/`/`figures/`/`results/` relative paths resolve, regardless of
   where you moved the notebook.
2. **Run once (cached)** — loads/generates the data (10 CSV replicates for
   simulated scenarios; the real-data file for the applications), then runs
   `BMA_SMC2` on **replicate 0** (edit `replicate_idx` for a different one)
   and caches the raw results to `results/.../*.pkl`. Re-running this cell
   later just reloads the cache instead of refitting — see "Restyling
   plots" below to force a refit.
3–6. **One cell per figure** — model weight, parameter estimates,
   likelihood, and the incidence/R_t fit-vs-forecast panel. Each cell takes
   a `style=` dict (colors, legend labels, font sizes, `ylim`, ...) you can
   edit and re-run freely, with no SMC² recomputation, exactly like you'd
   tweak a plot in your original notebooks.

For simulated scenarios, the parameter-estimate and fit plots overlay the
true value / true R_t, since the ground truth is known. Real-data
applications skip that overlay (there's no ground truth to compare against).

Prefer a plain script (e.g. for a batch job)? Every `.py` config file also
has a `main()` that runs the config's `analyse_replicate(...)` end-to-end
with default styling:

```bash
python -m simulation_study.scenario_A     # from the repo root
python -m real_data_study.flu_application
```

## Restyling plots without re-running SMC²

BMA-SMC² is the slow part; plotting is not. Every scenario/application
config accepts a `CACHE_PATH` (under `results/`) -- the first run computes
and pickles the raw `smc2_results`, and every later run loads that pickle
instead of recomputing (this is exactly what cell 2 of each notebook does).
On top of that, every `plot_*` function in `src/scenario_runner.py` takes a
`style=` dict (colors, legend labels, font sizes, axis limits -- see
`DEFAULT_STYLE` in that file), so restyling a figure is just editing the
dict in its cell and re-running that one cell -- no SMC² recomputation, and
you can also reload a cached run fresh in a new session:

```python
from src.scenario_runner import load_smc2_results, plot_model_weight

results = load_smc2_results("results/simulated/scenario_A/rep0.pkl")

plot_model_weight(
    results, forecast_days=21, title="Scenario A",
    figures_dir="figures/simulated/scenario_A",
    style={"color_dthp": "crimson", "color_seir": "navy", "ylim_weight": (0, 1)},
)
```

See `results/README.md` for where caches live and how to force a fresh run
(delete the `.pkl`, or pass `force_rerun=True`).

## Notes on renamed / relocated code

- `smc_squared.py` → `src/bma_smc2.py` (matches the function name `BMA_SMC2`)
- `smc.py` → `src/particle_filter.py` (its one function is `Particle_Filter`;
  the old name was easy to confuse with `smc_squared.py`)
- `smc_visualization.py` → `src/visualization.py`
- `ssm_prior_draw.py` → `src/priors.py`
- `observation_dist.py` → `src/observation_models.py`
- `models.py` → `src/epidemic_models.py`
- Renamed every `sir`/`SIR` identifier to `seir`/`SEIR` throughout the
  codebase (function names, variable names, dict keys, plot labels) since
  the actual models fit are SEIR/SEIRS (they have an exposed compartment),
  not SIR/SIRS. `stochastic_sir_model` → `stochastic_seir_model`,
  `stochastic_sirs_model` → `stochastic_seirs_model`, `model_sir` →
  `model_seir`, `loglik_sir` → `loglik_seir`, etc.
- `compute_model_average`, `compute_window_weights` (the model-weight
  calculation used to be copy-pasted inline in every notebook), and
  `extend_array` were consolidated into the new `src/helpers.py`.
- The "run BMA_SMC2, then plot the model weight / parameter estimates /
  likelihood / fit" logic that was duplicated (with small variations) across
  every scenario notebook is now the single `analyse_replicate` function in
  `src/scenario_runner.py`, imported by every scenario/application script.

## Known quirks carried over from the original code (worth a look before publishing)

- `PMMH_kernel`'s `log_prior` computes a Jacobian adjustment per parameter
  inside a loop but only adds the *last* parameter's adjustment to the
  total — kept as-is for numerical fidelity with the original results, but
  worth double-checking against the paper's derivation.
- `Particle_Filter`'s trajectory-storage branch (`add=1`) previously hard-coded
  `n_jobs=10` regardless of the `n_jobs` argument; this was fixed in
  `src/particle_filter.py` to respect the passed-in value.
