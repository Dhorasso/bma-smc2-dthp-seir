# Particle Filter Implementation
# ================================
# This script implements a parallele Particle Filter for state estimation 
# and marginal log-likelihood computation in the DTHP andd SIR model.


import numpy as np
import pandas as pd
import gc
from state_process import state_transition
from resampling import resampling_style
from joblib import Parallel, delayed  


def Parallele_Particle_Filter(
    model, model_type, state_names, current_state_particles, 
    theta, theta_names, observed_data, num_state_particles,
    observation_distribution, resampling_method='stratified',
    add=0, end=False, forecast_days=0, time=0
):
    """
    Perform Particle Filtering for either the DTHP or SIR model to estimate state evolution
    and compute the marginal log-likelihood.

   Parameters:
    - model (func): The model function.
    - model_type (str) : Type of model ('dthp' or 'SIR').
    - state_names (list): Names of the state variables.
    - current_state_particles (ndarray): The initial state particles.
    - theta (ndarray): Model parameters.
    - theta_names (list): Names of the theta parameters.
    - num_state_particles (int): Number of particles.
    - observation_distribution (func): Type of likelihood function.
    - resampling_method (str): Resampling method ('stratified', 'systematic', etc.).
    - observation_distribution (str): Type of observation distribution 
    - add (int): Flag to indicate whether to store state history.
    - end (bool): Flag to control trajectory addition.
    - forecast_days (int): Number of forecast days after observed data ends.

    Returns:
    dict
        Dictionary containing marginal log-likelihood, particle states, state history, 
        and trajectory states, distinguished by model type.
    """
    particle_weights = np.ones(num_state_particles) / num_state_particles
    num_timesteps = len(observed_data)
    
    traj_state = [{key: [] for key in ['time'] + state_names} for _ in range(num_state_particles)]
    state_hist = [None] * num_timesteps
    marginal_log_likelihood = 0
    
    data_app = observed_data[['obs']].copy()
    
    for t in range(num_timesteps + forecast_days):
        t_start, t_end = (0, 0) if t == 0 else (t - 1, t)
        
        if t < num_timesteps:
            current_data_point = observed_data.iloc[t]
        elif t >= num_timesteps and model_type == 'dthp':
            sample = np.array([p[0] for p in current_state_particles])
            y_t = np.median(sample)
            data_app = pd.concat([data_app, pd.DataFrame([[y_t]], columns=['obs'])], ignore_index=True)
        
        if model_type == 'dthp':
            trajectories = model(current_state_particles, theta, state_names, theta_names, data_app, t)
        elif model_type == 'SIR':
            trajectories = solve_model(model, theta, current_state_particles, state_names, theta_names, t_start, t_end)
        else:
            raise ValueError("Unknown model type. Use 'dthp' or 'SIR'.")
        
        model_points = trajectories.to_numpy()
        
        if t < num_timesteps:
            weights = observation_distribution(current_data_point, trajectories, theta, theta_names)
        A = np.max(weights)
        weights_mod = np.ones_like(weights) if A < -1e2 else np.exp(weights - A)
        normalized_weights = weights_mod / np.sum(weights_mod)
        resampled_indices = resampling_style(normalized_weights, resampling_method)
        current_state_particles = model_points[resampled_indices]
        
        zt = max(np.mean(np.exp(weights)), 1e-12)  # Avoid division by zero
        marginal_log_likelihood += np.log(zt)
        
        if add == 1:
            if end:
                traj_state = Parallel(n_jobs=10)(
                    delayed(lambda traj, j: pd.DataFrame(
                        {'time': list(traj['time']) + [t], 
                         **{name: list(traj[name]) + [current_state_particles[j][i]] for i, name in enumerate(state_names)}}
                    ))(traj, j) 
                    for j, traj in enumerate(traj_state)
                )
            else:
                state_hist[t] = current_state_particles
        
        gc.collect()
    
    return {
        'margLogLike': marginal_log_likelihood, 
        'particle_state': current_state_particles, 
        'state_hist': state_hist, 
        'traj_state': traj_state
    }

