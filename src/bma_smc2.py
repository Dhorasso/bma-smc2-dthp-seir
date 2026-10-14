"""
Bayesian Model Averaging via SMC^2 (BMA-SMC^2).

Runs two SMC^2 samplers side by side -- one for a DTHP model, one for a
SEIR/SEIRS-family model -- sharing the same observed data stream. At each time
step both models are propagated, their theta particles are reweighted and
(when the ESS drops too low) rejuvenated with a PMMH kernel. The per-step
model evidence for each model is also tracked, which downstream code
(`src.helpers.compute_window_weights`) turns into a time-varying model
weight for the ensemble/model-averaging forecast.
"""

import gc
import math
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from src.priors import (
    initial_theta,
    initial_state,
    initial_one_state,
    untransform_theta,
)
from src.state_process import state_transition
from src.particle_filter import Particle_Filter
from src.pmmh import PMMH_kernel
from src.resampling import resampling_style


def BMA_SMC2(
    model_dthp, model_seir, initial_state_info_dthp, initial_theta_info_dthp,
    initial_state_info_seir, initial_theta_info_seir, observed_data, num_state_particles,
    num_theta_particles, observation_distribution, resampling_threshold=0.5,
    resampling_method='stratified', tw=1, pmmh_moves=5, c=0.5, n_jobs=10,
    forecast_days=0, N=5e5, show_progress=True
):
    """
    Fit the DTHP and SEIR/SEIRS models jointly with SMC^2 and estimate the
    posterior model evidence over time for model averaging.

    Parameters
    ----------
    model_dthp, model_seir : callables
        Stochastic transition functions for each model.
    initial_state_info_dthp, initial_theta_info_dthp,
    initial_state_info_seir, initial_theta_info_seir : dict
        Prior specification for each model's state and parameters.
    observed_data : DataFrame
        Data to fit, with an 'obs' column.
    num_state_particles, num_theta_particles : int
        Number of state and theta particles.
    observation_distribution : callable
        See `src.observation_models`.
    resampling_threshold : float
        ESS threshold (as a fraction of num_theta_particles) that triggers
        theta resampling + PMMH rejuvenation.
    resampling_method : str
        'stratified', 'systematic', 'residual', or 'multinomial'.
    tw : int
        (Reserved) refresh window for the model evidence.
    pmmh_moves : int
        Number of PMMH moves per rejuvenation step.
    c : float
        Covariance scaling factor for the PMMH proposal.
    n_jobs : int
        Requested number of workers (auto-capped to 75% of CPU count).
    forecast_days : int
        Number of days to forecast beyond the observed data.
    N : float
        Total population size.
    show_progress : bool
        Show a tqdm progress bar.

    Returns
    -------
    dict with, for each model (suffix '_dthp' / '_seir'):
        'loglik_*'      per-step model evidence
        'traj_theta_*'  parameter trajectories (list of per-particle DataFrames)
        'traj_state_*'  forecast state trajectories from the final PMMH step
        'ESS_theta_*'   effective sample size of the theta particles over time
    """
    num_timesteps = len(observed_data)

    Z_dthp, Z_seir = np.zeros(num_theta_particles), np.zeros(num_theta_particles)
    model_evid_dthp, model_evid_seir = np.zeros(num_timesteps), np.zeros(num_timesteps)
    likelihood_increment_dthp = np.ones(num_theta_particles)
    likelihood_increment_seir = np.ones(num_theta_particles)
    theta_weights_dthp = np.ones((num_theta_particles, num_timesteps)) / num_theta_particles
    theta_weights_seir = theta_weights_dthp.copy()
    ESS_theta_dthp, ESS_theta_seir = np.zeros(num_timesteps), np.zeros(num_timesteps)

    def initialize_particles(model_type, initial_state_info, initial_theta_info):
        theta_init = initial_theta(initial_theta_info, num_theta_particles)
        state_init = initial_state(initial_state_info, num_theta_particles, num_state_particles)
        return {
            'name': model_type,
            'current_theta': theta_init['currentThetaParticles'],
            'theta_names': theta_init['thetaName'],
            'current_state': state_init['currentStateParticles'],
            'state_names': state_init['stateName'],
        }

    dthp_data = initialize_particles('dthp', initial_state_info_dthp, initial_theta_info_dthp)
    seir_data = initialize_particles('seir', initial_state_info_seir, initial_theta_info_seir)

    traj_theta_dthp = [{key: [] for key in ['time'] + dthp_data['theta_names']} for _ in range(num_theta_particles)]
    traj_theta_seir = [{key: [] for key in ['time'] + seir_data['theta_names']} for _ in range(num_theta_particles)]
    traj_state_dthp = {}
    traj_state_seir = {}

    n_jobs = max(4, math.floor(os.cpu_count() * 0.75))

    if show_progress:
        progress_bar = tqdm(total=num_timesteps, desc="BMA-SMC^2 Progress")

    for t in range(num_timesteps):
        current_data_point = observed_data.iloc[t]
        t_start, t_end = (0, 0) if t == 0 else (t - 1, t)

        def process_particle(model_type, theta_idx):
            model_data = dthp_data if model_type == 'dthp' else seir_data
            trans_theta = model_data['current_theta'][theta_idx]
            theta = untransform_theta(
                trans_theta,
                initial_theta_info_dthp if model_type == 'dthp' else initial_theta_info_seir,
            )
            state_particles = model_data['current_state'][theta_idx]

            if model_type == 'dthp':
                trajectories = model_dthp(
                    state_particles, theta, model_data['state_names'],
                    model_data['theta_names'], observed_data, t, N
                )
            else:
                trajectories = state_transition(
                    model_seir, theta, state_particles, model_data['state_names'],
                    model_data['theta_names'], t_start, t_end
                )
            model_points = trajectories.to_numpy()

            weights = observation_distribution(
                current_data_point, trajectories, theta, model_data['theta_names']
            )
            A = np.max(weights)
            weights_mod = np.ones_like(weights) if A < -1e2 else np.exp(weights - A)
            normalized_weights = weights_mod / np.sum(weights_mod)
            resampled_indices = resampling_style(normalized_weights, resampling_method)
            current_state_particles = model_points[resampled_indices]

            likelihood_increment_theta = max(np.mean(np.exp(weights)), 1e-12)
            return {
                'state_particles': current_state_particles,
                'likelihood': likelihood_increment_theta,
                'theta': trans_theta,
            }

        particles_dthp = Parallel(n_jobs=n_jobs)(
            delayed(process_particle)('dthp', m) for m in range(num_theta_particles)
        )
        particles_seir = Parallel(n_jobs=n_jobs)(
            delayed(process_particle)('seir', m) for m in range(num_theta_particles)
        )

        for (model_data, model, initial_state_info, initial_theta_info, theta_weights, Z, traj_theta,
             particles, likelihood_increment, model_evid, ESS_theta, traj_state) in [
            (dthp_data, model_dthp, initial_state_info_dthp, initial_theta_info_dthp, theta_weights_dthp,
             Z_dthp, traj_theta_dthp, particles_dthp, likelihood_increment_dthp, model_evid_dthp,
             ESS_theta_dthp, traj_state_dthp),
            (seir_data, model_seir, initial_state_info_seir, initial_theta_info_seir, theta_weights_seir,
             Z_seir, traj_theta_seir, particles_seir, likelihood_increment_seir, model_evid_seir,
             ESS_theta_seir, traj_state_seir),
        ]:
            model_data['current_state'] = np.array([p['state_particles'] for p in particles])
            model_data['current_theta'] = np.array([p['theta'] for p in particles])
            likelihood_increment[:] = np.array([p['likelihood'] for p in particles])

            Z += np.log(likelihood_increment)

            if t > 0:
                theta_weights[:, t] = theta_weights[:, t - 1] * likelihood_increment
                model_evid[t] = Evidence(theta_weights[:, t - 1], likelihood_increment)
            else:
                theta_weights[:, t] = theta_weights[:, t] * likelihood_increment
                model_evid[t] = Evidence(theta_weights[:, t], likelihood_increment)

            theta_weights[:, t] /= np.sum(theta_weights[:, t])
            ESS_theta[t] = 1 / np.sum(theta_weights[:, t] ** 2)

            # Resample + PMMH rejuvenation when the theta ESS drops too low
            if ESS_theta[t] < resampling_threshold * num_theta_particles:
                resampled_indices = resampling_style(theta_weights[:, t], resampling_method)
                Z = Z[resampled_indices]
                theta_mean = np.average(model_data['current_theta'], axis=0, weights=theta_weights[:, t])
                theta_covariance = np.cov(model_data['current_theta'].T, ddof=0, aweights=theta_weights[:, t])

                model_data['current_theta'] = model_data['current_theta'][resampled_indices]
                model_data['current_state'] = model_data['current_state'][resampled_indices]

                theta_weights[:, t] = np.ones(num_theta_particles) / num_theta_particles
                new_particles = Parallel(n_jobs=n_jobs)(delayed(PMMH_kernel)(
                    model, model_data['name'], Z[m], model_data['current_theta'], model_data['current_state'][m],
                    model_data['theta_names'], observed_data.iloc[:t + 1], model_data['state_names'],
                    initial_theta_info, initial_state_info, num_state_particles, theta_mean, theta_covariance,
                    observation_distribution, resampling_method, m, t, pmmh_moves, c, N=N, n_jobs=n_jobs
                ) for m in range(num_theta_particles))

                model_data['current_theta'] = np.array([new['theta'] for new in new_particles])
                model_data['current_state'] = np.array([new['state'] for new in new_particles])

            resampled_indices = resampling_style(theta_weights[:, t], resampling_method)
            theta_resample = model_data['current_theta'][resampled_indices]
            traj_theta = Parallel(n_jobs=n_jobs)(
                delayed(lambda traj, j: pd.DataFrame(
                    {'time': list(traj['time']) + [t],
                     **{name: list(traj[name]) + [untransform_theta(model_data['current_theta'][j], initial_theta_info)[i]]
                        for i, name in enumerate(model_data['theta_names'])}}
                ))(traj, j)
                for j, traj in enumerate(traj_theta)
            )
            if model_data['name'] == 'dthp':
                traj_theta_dthp = traj_theta
            else:
                traj_theta_seir = traj_theta

            # On the final step, run a full particle filter (with forecast) for a
            # small representative subset of theta draws, used for plotting/forecasting.
            if t == num_timesteps - 1:
                ini_state = initial_one_state(initial_state_info, num_state_particles)
                current_state = np.array(ini_state['currentStateParticles'])

                theta_mean = np.median(model_data['current_theta'], axis=0)
                dists = np.linalg.norm(model_data['current_theta'] - theta_mean, axis=1)
                idx = np.argsort(dists)[:100]
                theta_samples = model_data['current_theta'][idx, :]

                def run_pf_for_theta(theta_raw, theta_id):
                    theta = untransform_theta(theta_raw, initial_theta_info)
                    PF_results = Particle_Filter(
                        model, model_data['name'], model_data['state_names'], current_state, theta,
                        model_data['theta_names'], observed_data, num_state_particles, observation_distribution,
                        resampling_method, forecast_days=forecast_days, N=N, add=1, end=True, n_jobs=1
                    )
                    return [df.assign(theta_id=theta_id) for df in PF_results['traj_state']]

                all_traj_states = Parallel(n_jobs=n_jobs)(
                    delayed(run_pf_for_theta)(theta_samples[i], i) for i in range(len(theta_samples))
                )
                traj_state_flat = [traj for sublist in all_traj_states for traj in sublist]

                if model_data['name'] == 'dthp':
                    traj_state_dthp = traj_state_flat
                else:
                    traj_state_seir = traj_state_flat

        if show_progress:
            progress_bar.update(1)

        gc.collect()

    if show_progress:
        progress_bar.close()

    return {
        'loglik_dthp': model_evid_dthp,
        'loglik_seir': model_evid_seir,
        'traj_theta_dthp': traj_theta_dthp,
        'traj_theta_seir': traj_theta_seir,
        'traj_state_dthp': traj_state_dthp,
        'traj_state_seir': traj_state_seir,
        'ESS_theta_dthp': ESS_theta_dthp,
        'ESS_theta_seir': ESS_theta_seir,
    }


def Evidence(theta_weights, likelihood_increment):
    """Model evidence increment at one time step (theta-weighted average likelihood)."""
    return np.average(likelihood_increment, weights=theta_weights)
