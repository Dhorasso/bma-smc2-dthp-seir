"""
Plotting utilities for SMC^2 output: turning particle trajectories into
credible-interval ribbon plots, with optional fit/forecast shading.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter


def trace_smc(traject):
    """
    Convert a list of per-particle DataFrames (one row per time step) into a
    dict of {variable_name: N_particles x T ndarray}.
    """
    matrix_dict = {}
    state_names = list(traject[0].columns[1:])

    for state in state_names:
        matrices = [df[state].values.reshape(1, -1) for df in traject]
        combined = np.concatenate(matrices, axis=1)
        matrix_dict[state] = combined.reshape(-1, traject[0].shape[0])

    return matrix_dict


def plot_smc(matrix, ax, separation_point=None, date=None, window=1,
             color='midnightblue', label=None, show_50ci=False):
    """
    Plot the posterior median and 95% (optionally also 50%) credible interval
    of a N_particles x T particle matrix, with the fit period drawn solid and
    the forecast period (>= separation_point) drawn dashed/lighter.

    Parameters
    ----------
    matrix : ndarray, shape (N_particles, T)
    ax : matplotlib.axes.Axes
    separation_point : the fit/forecast cutoff (in the same units as `date`
        or as the integer time index if `date` is None)
    date : array-like of dates, or None to use an integer time axis
    window : rolling-average smoothing window
    show_50ci : also draw the 50% credible interval
    """
    median = np.nanmean(matrix, axis=0)
    ci_95 = np.nanpercentile(matrix, [2.5, 97.5], axis=0)
    if show_50ci:
        ci_50 = np.nanpercentile(matrix, [25, 75], axis=0)

    median = pd.Series(median).rolling(window=window, min_periods=1).mean().values
    ci_95 = pd.DataFrame(ci_95).T.rolling(window=window, min_periods=1).mean()
    if show_50ci:
        ci_50 = pd.DataFrame(ci_50).T.rolling(window=window, min_periods=1).mean()

    T = matrix.shape[1]
    time_steps = pd.to_datetime(date) if date is not None else np.arange(T)

    if separation_point is not None:
        condition = time_steps >= separation_point
        fit_times, pred_times = time_steps[~condition], time_steps[condition]

        ax.fill_between(fit_times, ci_95[0][~condition], ci_95[1][~condition], color=color, alpha=0.3)
        ax.fill_between(pred_times, ci_95[0][condition], ci_95[1][condition], color=color, alpha=0.12)
        if show_50ci:
            ax.fill_between(fit_times, ci_50[0][~condition], ci_50[1][~condition], color=color, alpha=0.4)
            ax.fill_between(pred_times, ci_50[0][condition], ci_50[1][condition], color=color, alpha=0.25)

        ax.plot(fit_times, median[~condition], color=color, lw=3, label=label)
        ax.plot(pred_times, median[condition], color=color, lw=3, linestyle='--', alpha=0.9)
        ax.axvline(separation_point, color='k', linestyle='--', lw=2)
    else:
        ax.fill_between(time_steps, ci_95.iloc[:, 0], ci_95.iloc[:, 1], color=color, alpha=0.3)
        if show_50ci:
            ax.fill_between(time_steps, ci_50.iloc[:, 0], ci_50.iloc[:, 1], color=color, alpha=0.5)
        ax.plot(time_steps, median, color=color, lw=3, label=label)

    if date is not None:
        ax.xaxis.set_major_locator(MonthLocator(interval=3))
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%b %y'))

    if label:
        ax.legend()


def corrected_matrix(matrix):
    """Replace per-column outliers (outside 1.5*IQR) with the column mean."""
    q1 = np.percentile(matrix, 25, axis=0)
    q3 = np.percentile(matrix, 75, axis=0)
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    for col in range(matrix.shape[1]):
        col_vals = matrix[:, col]
        mask = (col_vals >= lower_bound[col]) & (col_vals <= upper_bound[col])
        col_mean = np.mean(col_vals[mask])
        matrix[col_vals > upper_bound[col], col] = col_mean
        matrix[col_vals < lower_bound[col], col] = col_mean

    return matrix
