"""
Particle filter for a single model (DTHP or SEIR/SEIRS family), used to estimate
the state trajectory and the marginal log-likelihood for a fixed theta.
Called directly inside the PMMH rejuvenation step, and used by BMA_SMC2 at the
final time step to generate forecast trajectories for a subset of theta draws.
"""

import gc

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.state_process import state_transition
from src.resampling import resampling_style


def Particle_Filter(
    model, model_type, state_names, current_state_particles,
    theta, theta_names, observed_data, num_state_particles,
    observation_distribution, resampling_method='stratified',
    N=5e5, add=0, end=False, forecast_days=0, time=0, n_jobs=10
):
    """
    Run a particle filter for either the DTHP or SEIR/SEIRS model.

    Parameters
    ----------
    model : callable
        The model transition function.
    model_type : str
        'dthp' or 'seir'.
    state_names : list of str
        Names of the state variables.
    current_state_particles : ndarray
        Initial state particles.
    theta, theta_names : ndarray, list of str
        Model parameters and their names.
    observed_data : DataFrame
        Observed data with an 'obs' column.
    num_state_particles : int
        Number of particles.
    observation_distribution : callable
        Observation likelihood, see `src.observation_models`.
    resampling_method : str
        'stratified', 'systematic', 'residual', or 'multinomial'.
    N : float
        Total population size.
    add : int
        If 1, store the full particle trajectory history.
    forecast_days : int
        Number of forecast steps beyond the observed data.
    n_jobs : int
        Number of workers used when `add=1`.

    Returns
    -------
    dict with 'margLogLike', 'particle_state', and 'traj_state'.
    """
    num_timesteps = len(observed_data)
    traj_state = [{key: [] for key in ['time'] + state_names} for _ in range(num_state_particles)]
    marginal_log_likelihood = 0
    data_app = observed_data[['obs']].copy()

    for t in range(num_timesteps + forecast_days):
        t_start, t_end = (0, 0) if t == 0 else (t - 1, t)

        if t < num_timesteps:
            current_data_point = observed_data.iloc[t]
        elif model_type == 'dthp':
            y_t = np.mean([p[0] for p in current_state_particles])
            data_app = pd.concat([data_app, pd.DataFrame([[y_t]], columns=['obs'])], ignore_index=True)

        if model_type == 'dthp':
            trajectories = model(current_state_particles, theta, state_names, theta_names, data_app, t, N)
        elif model_type == 'seir':
            trajectories = state_transition(model, theta, current_state_particles, state_names, theta_names, t_start, t_end)
        else:
            raise ValueError("Unknown model type. Use 'dthp' or 'seir'.")

        if t < num_timesteps:
            weights = observation_distribution(current_data_point, trajectories, theta, theta_names)
            A = np.max(weights)
            weights_mod = np.ones_like(weights) if A < -1e3 else np.exp(weights - A)
            normalized_weights = weights_mod / np.sum(weights_mod)
            zt = max(np.mean(np.exp(weights)), 1e-12)
            marginal_log_likelihood += np.log(zt)
        else:
            trajectories = observation_distribution(current_data_point, trajectories, theta, theta_names, pred=True)

        model_points = trajectories.to_numpy()
        resampled_indices = resampling_style(normalized_weights, resampling_method)
        current_state_particles = model_points[resampled_indices]

        if add == 1:
            traj_state = Parallel(n_jobs=n_jobs)(
                delayed(lambda traj, j: pd.DataFrame(
                    {'time': list(traj['time']) + [t],
                     **{name: list(traj[name]) + [current_state_particles[j][i]] for i, name in enumerate(state_names)}}
                ))(traj, j)
                for j, traj in enumerate(traj_state)
            )

        gc.collect()

    return {
        'margLogLike': marginal_log_likelihood,
        'particle_state': current_state_particles,
        'traj_state': traj_state,
    }
