"""
Shared "run BMA-SMC^2 on one replicate, then plot" logic used by every
simulation-study scenario script (and, with real data, by the real-data
application scripts). Keeping this in one place means each scenario script
only has to define what's actually scenario-specific: the data, the priors,
and (for simulated data) the true parameter values.

Running BMA-SMC^2 is the slow part. To let you restyle a figure (colors,
labels, ylim, font sizes, ...) without re-running it:

1. Run once with a `cache_path`, e.g.:
       results = run_bma_smc2(..., cache_path="results/scenario_A_rep0.pkl")
   (or pass `cache_path=...` to `analyse_replicate`, which does the same).
2. In a later session / notebook cell, reload the cached results and call
   the plot_* functions directly with a `style` dict:
       from src.scenario_runner import load_smc2_results, plot_model_weight
       results = load_smc2_results("results/scenario_A_rep0.pkl")
       plot_model_weight(results, forecast_days=21, title="Scenario A",
                          figures_dir="figures/simulated/scenario_A",
                          style={"color_dthp": "crimson", "ylim_weight": (0, 1)})
   No SMC^2 computation happens in step 2 -- only plotting.

See `DEFAULT_STYLE` below for every customisable key (colors, labels, font
sizes, ylim). Pass a partial dict to `style=`; anything you don't override
keeps its default.
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.bma_smc2 import BMA_SMC2
from src.helpers import compute_window_weights, compute_model_average
from src.visualization import trace_smc, plot_smc

# Sliding-window model-weight settings used throughout the paper's figures.
DEFAULT_WINDOW = 25
DEFAULT_T_BURNING = 0

# Every plot_* function below merges its `style=` argument on top of this.
# Pass e.g. style={"color_dthp": "crimson", "ylim_weight": (0, 1)} to override
# just the keys you care about.
DEFAULT_STYLE = {
    # colors
    "color_dthp": "orange",
    "color_seir": "dodgerblue",
    "color_ma": "green",
    "color_true": "black",
    "color_obs_fit": "black",
    "color_obs_forecast": "brown",
    # legend labels
    "label_dthp": "DTHP",
    "label_seir": "SEIR",
    "label_ma": "MA",
    # font sizes
    "fontsize_title": 18,
    "fontsize_label": 16,
    "fontsize_tick": 14,
    "fontsize_legend": 14,
    # axis limits (None = matplotlib auto)
    "ylim_weight": (-0.02, 1.05),
    "ylim_likelihood": None,
    "ylim_incidence": None,
    "ylim_rt": None,
    "ylim_param": None,  # applied to every parameter subplot if set
}


def _style(overrides):
    return {**DEFAULT_STYLE, **(overrides or {})}


# ---------------------------------------------------------------------------
# Running (+ caching) BMA-SMC^2
# ---------------------------------------------------------------------------

def save_smc2_results(smc2_results, path):
    """Pickle a `BMA_SMC2` results dict to `path` (parent dirs auto-created)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(smc2_results, f)
    print(f"Results cached to: {path}")


def load_smc2_results(path):
    """Load a `BMA_SMC2` results dict previously saved with `save_smc2_results`."""
    with open(path, "rb") as f:
        return pickle.load(f)


def run_bma_smc2(
    model_seir, model_dthp, state_info_seir, theta_info_seir,
    state_info_dthp, theta_info_dthp, observed_data, forecast_days, N_pop,
    observation_distribution, num_state_particles=200, num_theta_particles=400,
    seed=123, cache_path=None, force_rerun=False,
):
    """
    Thin wrapper around `BMA_SMC2` that fixes the seed for reproducibility.

    If `cache_path` is given and already exists, the cached results are
    loaded and BMA-SMC^2 is *not* re-run (unless `force_rerun=True`). If
    `cache_path` is given and doesn't exist yet, results are computed then
    saved there for next time.
    """
    if cache_path and os.path.exists(cache_path) and not force_rerun:
        print(f"Loading cached BMA-SMC^2 results from: {cache_path}")
        return load_smc2_results(cache_path)

    np.random.seed(seed)
    results = BMA_SMC2(
        model_seir=model_seir,
        model_dthp=model_dthp,
        initial_state_info_dthp=state_info_dthp,
        initial_theta_info_dthp=theta_info_dthp,
        initial_state_info_seir=state_info_seir,
        initial_theta_info_seir=theta_info_seir,
        observed_data=observed_data,
        num_state_particles=num_state_particles,
        num_theta_particles=num_theta_particles,
        observation_distribution=observation_distribution,
        forecast_days=forecast_days,
        N=N_pop,
    )

    if cache_path:
        save_smc2_results(results, cache_path)

    return results


# ---------------------------------------------------------------------------
# Plotting (pure functions of `smc2_results` -- no SMC^2 computation here,
# so these can always be re-run cheaply with a different `style`)
# ---------------------------------------------------------------------------

def plot_model_weight(
    smc2_results, forecast_days, title, figures_dir, filename="model_weight.pdf",
    window=DEFAULT_WINDOW, t_burning=DEFAULT_T_BURNING, eps=1e-12, style=None,
):
    """Plot the DTHP vs SEIR posterior model weight over time, save as PDF."""
    s = _style(style)
    lik_dthp = np.array(smc2_results['loglik_dthp'])
    lik_seir = np.array(smc2_results['loglik_seir'])
    loglik_dthp = np.log(lik_dthp + eps)
    loglik_seir = np.log(lik_seir + eps)

    w_dthp, w_seir = compute_window_weights(loglik_dthp, loglik_seir, window=window, t_burning=t_burning)

    T = len(w_seir)
    w_dthp = np.concatenate([w_dthp, np.repeat(w_dthp[-1], forecast_days)])
    w_seir = np.concatenate([w_seir, np.repeat(w_seir[-1], forecast_days)])
    x = np.arange(len(w_seir))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, w_seir, color=s["color_seir"], lw=2.2, label=s["label_seir"])
    ax.plot(x, w_dthp, color=s["color_dthp"], lw=2.2, label=s["label_dthp"])
    ax.axhline(y=0.5, color='gray', linestyle='--', lw=1.6, alpha=0.75)
    ax.axvline(x=T - 1, color='black', linestyle='--', lw=1.8, label="Forecast start")
    ax.set_title(title, fontsize=s["fontsize_title"], fontweight='bold', pad=12)
    ax.set_xlabel("Days", fontsize=s["fontsize_label"], fontweight='bold')
    ax.set_ylabel("Model weight", fontsize=s["fontsize_label"], fontweight='bold')
    if s["ylim_weight"] is not None:
        ax.set_ylim(*s["ylim_weight"])
    ax.set_xlim(0, len(x) - 1)
    ax.tick_params(axis='both', labelsize=s["fontsize_tick"])
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=s["fontsize_legend"], frameon=True, loc='upper right')
    plt.tight_layout()

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, filename)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.show()
    print(f"Figure saved to: {out_path}")

    return w_dthp, w_seir


def plot_likelihood(
    smc2_results, title, figures_dir, filename="likelihood.pdf", eps=1e-12, style=None,
):
    """Plot the per-step model evidence (log scale) for DTHP vs SEIR, save as PDF."""
    s = _style(style)
    lik_dthp = np.array(smc2_results['loglik_dthp'])
    lik_seir = np.array(smc2_results['loglik_seir'])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(np.log(lik_seir + eps), color=s["color_seir"], lw=2, label=s["label_seir"])
    ax.plot(np.log(lik_dthp + eps), color=s["color_dthp"], lw=2, label=s["label_dthp"])
    ax.set_title(title, fontsize=s["fontsize_title"], fontweight='bold', pad=12)
    ax.set_xlabel("Days", fontsize=s["fontsize_label"], fontweight='bold')
    ax.set_ylabel("Log model evidence (per step)", fontsize=s["fontsize_label"], fontweight='bold')
    if s["ylim_likelihood"] is not None:
        ax.set_ylim(*s["ylim_likelihood"])
    ax.tick_params(axis='both', labelsize=s["fontsize_tick"])
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=s["fontsize_legend"], frameon=True)
    plt.tight_layout()

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, filename)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.show()
    print(f"Figure saved to: {out_path}")


def plot_param_estimates(
    smc2_results, title, figures_dir, filename="param_estimates.pdf",
    param_labels=None, true_values=None, ncols=3, style=None,
):
    """
    Plot the posterior trajectory (median + 50%/95% CrI) of every DTHP and
    SEIR parameter, overlaying a dashed true-value line for any parameter
    present in `true_values`.

    Parameters
    ----------
    param_labels : dict[str, str], optional
        Maps parameter name -> display label (e.g. {'gamma': r'$\\gamma$'}).
        Defaults to the raw parameter name.
    true_values : dict[str, float], optional
        Maps parameter name -> true value to overlay (only used for
        simulated-data scenarios where the ground truth is known).
    style : dict, optional
        See `DEFAULT_STYLE`; `ylim_param` (applied to every subplot),
        `color_dthp`/`color_seir`, and the font-size keys are used here.
    """
    s = _style(style)
    param_labels = param_labels or {}
    true_values = true_values or {}

    matrix_dict_theta_dthp = trace_smc(smc2_results['traj_theta_dthp'])
    matrix_dict_theta_seir = trace_smc(smc2_results['traj_theta_seir'])

    all_params = [('dthp', k, v) for k, v in matrix_dict_theta_dthp.items()] + \
                 [('seir', k, v) for k, v in matrix_dict_theta_seir.items()]

    n_plots = len(all_params)
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (model_type, name, matrix) in zip(axes, all_params):
        plot_smc(matrix, ax=ax, show_50ci=True,
                 color=s["color_dthp"] if model_type == 'dthp' else s["color_seir"])
        ax.set_ylabel(param_labels.get(name, name), fontsize=s["fontsize_label"] + 4, fontweight='bold')
        ax.set_xlabel("Days", fontsize=s["fontsize_label"] - 2)
        if s["ylim_param"] is not None:
            ax.set_ylim(*s["ylim_param"])
        ax.tick_params(axis='both', labelsize=s["fontsize_tick"])
        ax.grid(True, linestyle='--', alpha=0.6)

        if name in true_values:
            ax.axhline(true_values[name], color=s["color_true"], linestyle='--', linewidth=2.5, label='True value')
            ax.legend(fontsize=s["fontsize_legend"] - 3)

    for ax in axes[n_plots:]:
        fig.delaxes(ax)

    fig.suptitle(title, fontsize=s["fontsize_title"], fontweight='bold')
    plt.tight_layout()

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, filename)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.show()
    print(f"Figure saved to: {out_path}")


def plot_fit(
    smc2_results, full_data, w_dthp, w_seir, separation_point, forecast_end,
    title, figures_dir, filename="fit.pdf", time_col="time", date_axis=False,
    true_rt_col=None, window=1, style=None,
):
    """
    Plot the incidence and Rt fit/forecast: DTHP + SEIR overlaid in the left
    column, the model-averaged (MA) ensemble in the right column, with
    observed data (fit period in `color_obs_fit`, forecast period in
    `color_obs_forecast`) and the forecast region shaded. If `true_rt_col`
    is given (simulated-data scenarios only), the true Rt trajectory is
    overlaid on both Rt panels.

    `w_dthp`/`w_seir` must already be extended to cover the forecast horizon
    (as returned by `plot_model_weight`).
    """
    s = _style(style)
    matrix_dict_state_dthp = trace_smc(smc2_results['traj_state_dthp'])
    matrix_dict_state_seir = trace_smc(smc2_results['traj_state_seir'])
    matrix_dict_state_avg = compute_model_average(matrix_dict_state_dthp, matrix_dict_state_seir, w_dthp, w_seir)

    time = pd.to_datetime(full_data[time_col]) if date_axis else full_data[time_col]
    date_arg = time if date_axis else None
    rolling_obs = full_data['obs'].rolling(window=window, min_periods=1).mean()
    before_sep = time < (pd.to_datetime(separation_point) if date_axis else separation_point)
    after_sep = ~before_sep

    fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharex=True, sharey='row')

    for ax in axs.flat:
        ax.axvspan(separation_point, forecast_end, facecolor='lightgray', alpha=0.25, zorder=0)

    def plot_incidence(ax, series, legend_loc='upper left'):
        for matrix_dict, label, color in series:
            plot_smc(matrix_dict['NI'], ax=ax, separation_point=separation_point,
                     date=date_arg, window=window, color=color, label=label)
        ax.scatter(time[before_sep], rolling_obs[before_sep], facecolor=s["color_obs_fit"],
                   marker='*', s=70, label='Observed (fit)', zorder=4)
        ax.scatter(time[after_sep], rolling_obs[after_sep], facecolor=s["color_obs_forecast"],
                   marker='*', s=70, label='Observed (forecast)', zorder=4)
        ax.axvline(separation_point, color='black', ls='--', lw=2)
        if s["ylim_incidence"] is not None:
            ax.set_ylim(*s["ylim_incidence"])
        ax.tick_params(labelsize=s["fontsize_tick"])
        ax.legend(fontsize=s["fontsize_legend"] - 1, loc=legend_loc, frameon=False)

    def plot_rt(ax, series):
        for matrix_dict, label, color in series:
            plot_smc(matrix_dict['Rt'], ax=ax, separation_point=separation_point,
                     date=date_arg, window=window, color=color, label=label)
        if true_rt_col is not None:
            true_rt = full_data[true_rt_col].rolling(window=window, min_periods=1).mean()
            ax.plot(time, true_rt, color=s["color_obs_forecast"], lw=3, zorder=3, label='True $R_t$')
        ax.axhline(1, color='k', ls='--', lw=2)
        if s["ylim_rt"] is not None:
            ax.set_ylim(*s["ylim_rt"])
        ax.tick_params(labelsize=s["fontsize_tick"])
        ax.legend(fontsize=s["fontsize_legend"] - 1, loc='upper right', frameon=False)

    dthp_seir_series = [
        (matrix_dict_state_dthp, s["label_dthp"], s["color_dthp"]),
        (matrix_dict_state_seir, s["label_seir"], s["color_seir"]),
    ]
    ma_series = [(matrix_dict_state_avg, s["label_ma"], s["color_ma"])]

    plot_incidence(axs[0, 0], dthp_seir_series)
    axs[0, 0].set_ylabel("Incidence", fontsize=s["fontsize_label"] + 4, fontweight='bold')

    plot_incidence(axs[0, 1], ma_series)

    plot_rt(axs[1, 0], dthp_seir_series)
    axs[1, 0].set_ylabel(r"Reproduction number $R_t$", fontsize=s["fontsize_label"] + 4, fontweight='bold')

    plot_rt(axs[1, 1], ma_series)

    for ax in axs.flat:
        ax.grid(True, ls='--', alpha=0.5)
    fig.suptitle(title, fontsize=s["fontsize_title"], fontweight='bold')
    plt.tight_layout()

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, filename)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.show()
    print(f"Figure saved to: {out_path}")


# ---------------------------------------------------------------------------
# Convenience wrapper: run once (with caching) + produce all four figures
# ---------------------------------------------------------------------------

def analyse_replicate(
    scenario_name, model_seir, model_dthp, state_info_seir, theta_info_seir,
    state_info_dthp, theta_info_dthp, observed_data, forecast_days, N_pop,
    observation_distribution, figures_dir, param_labels=None, true_values=None,
    num_state_particles=200, num_theta_particles=400, seed=123,
    full_data=None, time_col="time", date_axis=False, true_rt_col=None,
    window=DEFAULT_WINDOW, t_burning=DEFAULT_T_BURNING,
    cache_path=None, force_rerun=False, style=None,
):
    """
    Run BMA-SMC^2 on one replicate/dataset (or load it from `cache_path` if
    already computed) and produce the four standard diagnostic figures:
    model weight, parameter estimates (vs. truth if known), per-model
    likelihood, and the incidence/Rt fit-vs-forecast panel. Figures are
    saved under `figures_dir`.

    `full_data` should be the *complete* dataset (fit + forecast horizon);
    if omitted, `observed_data` is used and no forecast-period observations
    are shown on the fit plot. `true_rt_col` (e.g. 'Rt') overlays the true
    reproduction number on the fit plot -- only meaningful for simulated
    scenarios where the ground truth is known.

    `cache_path`, if given, both loads a previous run's results (skipping
    BMA-SMC^2 entirely, unless `force_rerun=True`) and saves a fresh run's
    results for next time. `style` is a partial override of `DEFAULT_STYLE`,
    applied to all four figures (call the plot_* functions directly if you
    want different styling per figure).

    Returns the raw `smc2_results` dict in case further analysis is needed.
    """
    smc2_results = run_bma_smc2(
        model_seir, model_dthp, state_info_seir, theta_info_seir,
        state_info_dthp, theta_info_dthp, observed_data, forecast_days, N_pop,
        observation_distribution, num_state_particles, num_theta_particles,
        seed=seed, cache_path=cache_path, force_rerun=force_rerun,
    )

    w_dthp, w_seir = plot_model_weight(
        smc2_results, forecast_days, title=scenario_name, figures_dir=figures_dir,
        window=window, t_burning=t_burning, style=style,
    )
    plot_param_estimates(smc2_results, title=scenario_name, figures_dir=figures_dir,
                          param_labels=param_labels, true_values=true_values, style=style)
    plot_likelihood(smc2_results, title=scenario_name, figures_dir=figures_dir, style=style)

    fit_data = full_data if full_data is not None else observed_data
    separation_point = fit_data[time_col].iloc[len(observed_data) - 1]
    forecast_end = fit_data[time_col].iloc[-1]
    plot_fit(
        smc2_results, fit_data, w_dthp, w_seir, separation_point, forecast_end,
        title=scenario_name, figures_dir=figures_dir, time_col=time_col,
        date_axis=date_axis, true_rt_col=true_rt_col, style=style,
    )

    return smc2_results
