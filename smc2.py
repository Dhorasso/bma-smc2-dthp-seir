
import numpy as np
import pandas as pd
import gc
from joblib import Parallel, delayed
from tqdm import tqdm
from ssm_prior_draw import*
from state_process import state_transition
from smc import Particle_Filter
from pmmh import PMMH_kernel
from resampling import resampling_style
#from observation_dist import compute_log_weight


def BMA_SMC2(
    model_dthp, model_seir, initial_state_info_dthp, initial_theta_info_dthp,
    initial_state_info_seir, initial_theta_info_seir, observed_data, num_state_particles,
    num_theta_particles, resampling_threshold=0.5, resampling_method='stratified', 
    observation_distribution='normal_approx_NB', forecast_days=0, show_progress=True
):
    """
    Perform Sequential Monte Carlo squared (SMC^2) for two models (dthp and seir), estimating the state and 
    parameters using a particle filter approach, compute the posterior model probability

    Parameters:
    - model_dthp, model_seir: Models for particle filtering.
    - initial_state_info_dthp, initial_theta_info_dthp: Initialization for the dthp model.
    - initial_state_info_seir, initial_theta_info_seir: Initialization for the seir model.
    - observed_data: Data to fit the model to.
    - num_state_particles, num_theta_particles: Number of state and theta particles.
    - resampling_threshold: Effective sample size threshold for resampling.
    - resampling_method: Method used for resampling.
    - observation_distribution: Distribution for observation likelihood computation.
    - forecast_days: Number of days to forecast beyond observed data.
    - show_progress: Whether to display progress bar.

    Returns:
    - Dictionary with weights, state trajectories, parameter trajectories, ESS, and acceptance rates.
    """
    num_timesteps = len(observed_data)
    
    # Initialize arrays to store results
    Z_arr_dthp, Z_arr_seir = np.zeros((num_theta_particles, num_timesteps)), np.zeros((num_theta_particles, num_timesteps))
    model_evid_dthp, model_evid_seir = np.zeros(num_timesteps), np.zeros(num_timesteps)
    likelihood_increment_dthp, likelihood_increment_seir = np.ones(num_theta_particles), np.ones(num_theta_particles)
    theta_weights_dthp = theta_weights_seir = np.ones((num_theta_particles, num_timesteps)) / num_theta_particles
    ESS_theta_dthp, ESS_theta_seir = np.zeros(num_timesteps), np.zeros(num_timesteps)

    # Initialize state and theta particles for both models
    def initialize_particles(model_type, initial_state_info, initial_theta_info):
        theta_init = initial_theta(initial_theta_info, num_theta_particles)
        state_init = initial_state(initial_state_info, num_theta_particles, num_state_particles)
        return {
            'current_theta': theta_init['currentThetaParticles'],
            'theta_names': theta_init['thetaName'],
            'current_state': state_init['currentStateParticles'],
            'state_names': state_init['stateName'],
            'state_history': np.zeros((num_timesteps, num_theta_particles, num_state_particles, len(state_init['stateName'])))
        }
    
    dthp_data = initialize_particles('dthp', initial_state_info_dthp, initial_theta_info_dthp)
    seir_data = initialize_particles('seir', initial_state_info_seir, initial_theta_info_seir)
    
    # Initialize progress bar
    if show_progress:
        progress_bar = tqdm(total=num_timesteps, desc="SMC Progress")
    
    for t in range(num_timesteps):
        current_data_point = observed_data.iloc[t]
        
        def process_particle(model_type, theta_idx):
            model_data = dthp_data if model_type == 'dthp' else seir_data
            trans_theta = model_data['current_theta'][theta_idx]
            theta = untransform_theta(trans_theta, initial_theta_info_dthp if model_type == 'dthp' else initial_theta_info_seir)
            state_particles = model_data['current_state'][theta_idx]
            
            if model_type == 'dthp':
                trajectories = model_dthp(state_particles, theta, model_data['state_names'], model_data['theta_names'], observed_data, t)
            else:
                trajectories = solve_model(model_seir, theta, state_particles, model_data['state_names'], model_data['theta_names'], t-1, t)
            
            weights = compute_log_weight(current_data_point, trajectories, theta, model_data['theta_names'], observation_distribution)
            normalized_weights = np.exp(weights - np.max(weights)) / np.sum(np.exp(weights - np.max(weights)))
            resampled_indices = resampling_style(normalized_weights, resampling_method)
            
            return {
                'state_particles': trajectories.to_numpy()[resampled_indices],
                'likelihood': np.mean(np.exp(weights)),
                'theta': trans_theta,
            }
        
        particles_dthp = Parallel(n_jobs=10)(delayed(process_particle)('dthp', m) for m in range(num_theta_particles))
        particles_seir = Parallel(n_jobs=10)(delayed(process_particle)('seir', m) for m in range(num_theta_particles))
        
        # Update state and theta particles
        for model_data, particles, likelihood_increment, Z_arr, model_evid, theta_weights in (
            (dthp_data, particles_dthp, likelihood_increment_dthp, Z_arr_dthp, model_evid_dthp, theta_weights_dthp),
            (seir_data, particles_seir, likelihood_increment_seir, Z_arr_seir, model_evid_seir, theta_weights_seir),
        ):
            model_data['current_state'] = np.array([p['state_particles'] for p in particles])
            model_data['current_theta'] = np.array([p['theta'] for p in particles])
            likelihood_increment[:] = np.array([p['likelihood'] for p in particles])
            Z_arr[:, t] = np.log(likelihood_increment)
            
            if t > 0:
                theta_weights[:, t] = theta_weights[:, t-1] * likelihood_increment
                model_evid[t] = Evidence(theta_weights[:, t-1], likelihood_increment)
            else:
                theta_weights[:, t] = theta_weights[:, t] * likelihood_increment
                model_evid[t] = Evidence(theta_weights[:, t], likelihood_increment)
            
            theta_weights[:, t] /= np.sum(theta_weights[:, t])
            ESS_theta = ESS_theta_dthp if model_data is dthp_data else ESS_theta_seir
            ESS_theta[t] = 1 / np.sum(theta_weights[:, t] ** 2)
        
        # Resampling step
        for model_data, ESS_theta, Z_arr, theta_weights in (
            (dthp_data, ESS_theta_dthp, Z_arr_dthp, theta_weights_dthp),
            (seir_data, ESS_theta_seir, Z_arr_seir, theta_weights_seir),
        ):
            if ESS_theta[t] < resampling_threshold * num_theta_particles:
                resampled_indices = resampling_style(theta_weights[:, t], resampling_method)
                model_data['current_theta'] = model_data['current_theta'][resampled_indices]
                model_data['current_state'] = model_data['current_state'][resampled_indices]
                theta_weights[:, t] = np.ones(num_theta_particles) / num_theta_particles
