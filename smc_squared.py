
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
    model_dthp, model_sir, initial_state_info_dthp, initial_theta_info_dthp,
    initial_state_info_sir, initial_theta_info_sir, observed_data, num_state_particles,
    num_theta_particles, resampling_threshold=0.5, observation_distribution, 
    resampling_method='stratified', tw=1, pmmh_moves=5, c=0.5, n_jobs=10, 
    forecast_days=0, show_progress=True
):
    """
    Perform Sequential Monte Carlo squared (SMC^2) for two models (dthp and sir), estimating the state and 
    parameters using a particle filter approach, compute the posterior model probability

    Parameters:
    - model_dthp, model_sir: stochatic model.
    - initial_state_info_dthp, initial_theta_info_dthp: Initialization for the dthp model.
    - initial_state_info_sir, initial_theta_info_sir: Initialization for the sir model.
    - observed_data: Data to fit the model to.
    - num_state_particles, num_theta_particles: Number of state and theta particles.
    - resampling_threshold: Effective sample size threshold for resampling.
    - observation_distribution (func): Type of observation likelihood.
    - resampling_method (str): Method used for resampling ('stratified', 'systematic', 'residual', 'multinomial').
    - tw (int): Refreshing whindow for the model evidence.
    - pmmh_moves (int): Number of PMMH move in the rejuvenation step.
    - c (int): scaling factor for the covariance matrix in the PMMH kernel.
    - n_jobs (int): Number of processor in the PMMH parallel computing
    - forecast_days: Number of days to forecast beyond observed data.
    - show_progress: Whether to display progress bar.

    Returns:
    - Dictionary with modle weights, state trajectories, parameter trajectories, and ESS
    """
    num_timesteps = len(observed_data)
    
    # Initialize arrays to store results
    Z_arr_dthp, Z_arr_sir = np.zeros((num_theta_particles, num_timesteps)), np.zeros((num_theta_particles, num_timesteps))
    Z_dthp, Z_sir = np.zeros(num_theta_particles), np.zeros(num_theta_particles)
    model_evid_dthp, model_evid_sir = np.zeros(num_timesteps), np.zeros(num_timesteps)
    likelihood_increment_dthp, likelihood_increment_sir = np.ones(num_theta_particles), np.ones(num_theta_particles)
    theta_weights_dthp = np.ones((num_theta_particles, num_timesteps)) / num_theta_particles
    heta_weights_sir = theta_weights_dthp.copy()
    ESS_theta_dthp, ESS_theta_sir = np.zeros(num_timesteps), np.zeros(num_timesteps)

    # Initialize state and theta particles for both models
    def initialize_particles(model_type, initial_state_info, initial_theta_info):
        theta_init = initial_theta(initial_theta_info, num_theta_particles)
        state_init = initial_state(initial_state_info, num_theta_particles, num_state_particles)
        return {
            'name': model_type,
            'current_theta': theta_init['currentThetaParticles'],
            'theta_names': theta_init['thetaName'],
            'current_state': state_init['currentStateParticles'],
            'state_names': state_init['stateName'],
            'state_history': np.zeros((num_timesteps, num_theta_particles, num_state_particles, len(state_init['stateName'])))
        }
    
    dthp_data = initialize_particles('dthp', initial_state_info_dthp, initial_theta_info_dthp)
    sir_data = initialize_particles('sir', initial_state_info_sir, initial_theta_info_sir)

    # Initialize trajectory storage for both models
    traj_theta_dthp = [{key: [] for key in ['time'] + dthp_data['theta_names']} for _ in range(num_theta_particles)]
    traj_theta_sir = [{key: [] for key in ['time'] + sir_data['theta_names']} for _ in range(num_theta_particles)]

    traj_state_dthp ={}
    traj_state_sir ={}
    # Initialize progress bar
    if show_progress:
        progress_bar = tqdm(total=num_timesteps, desc="BMA-SMC^2 Progress")
    
    for t in range(num_timesteps):
        current_data_point = observed_data.iloc[t]
        t_start, t_end = (0, 0) if t == 0 else (t - 1, t)
        
        def process_particle(model_type, theta_idx):
            model_data = dthp_data if model_type == 'dthp' else sir_data
            trans_theta = model_data['current_theta'][theta_idx]
            theta = untransform_theta(trans_theta, initial_theta_info_dthp if model_type == 'dthp' else initial_theta_info_sir)
            state_particles = model_data['current_state'][theta_idx]
            
            if model_type == 'dthp':
                trajectories = model_dthp(state_particles, theta, model_data['state_names'], model_data['theta_names'], observed_data, t)
            else:
                trajectories = solve_model(model_sir, theta, state_particles, model_data['state_names'], model_data['theta_names'], t_start, t_end)
            model_points = trajectories.to_numpy()
            
            weights = observation_distribution(current_data_point, trajectories, theta, model_data['theta_names'])
            
            A = np.max(weights)
            weights_mod = np.ones_like(weights) if A < -1e2 else np.exp(weights - A)
            normalized_weights = weights_mod / np.sum(weights_mod)
            resampled_indices = resampling_style(normalized_weights, resampling_method)
            current_state_particles = model_points[resampled_indices]

            # Likelihood increment for this particle
            likelihood_increment_theta = max(np.mean(np.exp(weights)), 1e-12)
            return {
                'state_particles': current_state_particles,
                'likelihood': likelihood_increment_theta,
                'theta': trans_theta,
            }
        
        particles_dthp = Parallel(n_jobs=n_jobs)(delayed(process_particle)('dthp', m) for m in range(num_theta_particles))
        particles_sir = Parallel(n_jobs=n_jobs)(delayed(process_particle)('sir', m) for m in range(num_theta_particles))
        
        # Update state and theta particles
        for model_data, model, initial_state_info, initial_theta_info, theta_weights, Z, state_names, traj_theta, particles, likelihood_increment, Z_arr, model_evid, ESS_theta, traj_state in [
            (dthp_data, model_dthp, initial_state_info_dthp, initial_theta_info_dthp, theta_weights_dthp, Z_dthp, dthp_data['state_names'], traj_theta_dthp,
             particles_dthp, likelihood_increment_dthp, Z_arr_dthp, model_evid_dthp, ESS_theta_dthp, traj_state_dthp),
            (sir_data, model_sir, initial_state_info_sir, initial_theta_info_sir, theta_weights_sir, Z_sir, sir_data['state_names'], traj_theta_sir,
             particles_sir, likelihood_increment_sir, Z_arr_sir, model_evid_sir, ESS_theta_sir, traj_state_sir)
        ]:
            model_data['current_state'] = np.array([p['state_particles'] for p in particles])
            model_data['current_theta'] = np.array([p['theta'] for p in particles])
            model_data['state_history'][t] =  model_data['current_state']
            likelihood_increment[:] = np.array([p['likelihood'] for p in particles])
            Z_arr[:, t] = np.log(likelihood_increment)
            Z = np.sum(Z_arr[:, :t + 1], axis=1)
            
            if t > 0:
                theta_weights[:, t] = theta_weights[:, t-1] * likelihood_increment
                model_evid[t] = Evidence(theta_weights[:, t-1], likelihood_increment)
            else:
                theta_weights[:, t] = theta_weights[:, t] * likelihood_increment
                model_evid[t] = Evidence(theta_weights[:, t], likelihood_increment)
            
            theta_weights[:, t] /= np.sum(theta_weights[:, t])
            ESS_theta[t] = 1 / np.sum(theta_weights[:, t] ** 2)
        
            # Resampling step
            if ESS_theta[t] < resampling_threshold * num_theta_particles:
                resampled_indices = resampling_style(theta_weights[:, t], resampling_method)
                Z = Z[resampled_indices]
                # Calculate the weighted mean and covariance for theta
                theta_mean = np.average(model_data['current_theta'], axis=0, weights=theta_weights[:, t])
                theta_covariance = np.cov(model_data['current_theta'].T, ddof=0, aweights=theta_weights[:, t])
                
                # Resample the data
                model_data['current_theta'] = model_data['current_theta'][resampled_indices]
                model_data['current_state'] = model_data['current_state'][resampled_indices]
                
                # Reset weights and run the PMMH kernel
                theta_weights[:, t] = np.ones(num_theta_particles) / num_theta_particles
                new_particles = Parallel(n_jobs=n_jobs)(delayed(PMH_kernel)(
                    model, model_data['name'], Z[m], model_data['current_theta'], model_data['state_history'], model_data['theta_names'],
                    observed_data.iloc[:t + 1], model_data['state_names'], initial_theta_info, num_state_particles, theta_mean,
                    theta_covariance,  observation_distribution, resampling_method, m, t, pmmh_moves, c) for m in range(num_theta_particles))
        
                # Update particles and states
                model_data['current_theta'] = np.array([new['theta'] for new in new_particles])
                model_data['current_state'] = np.array([new['state'] for new in new_particles])
                Z = np.array([new['Z'] for new in new_particles])

        
            # Update the trajectory of theta over time
            traj_theta = Parallel(n_jobs=n_jobs)(
                delayed(lambda traj, j: pd.DataFrame(
                    {'time': list(traj['time']) + [t], 
                     **{name: list(traj[name]) + [untransform_theta(model_data['current_theta'][j], initial_theta_info)[i]] 
                        for i, name in enumerate(model_data['theta_names'])}}
                ))(traj, j) 
                for j, traj in enumerate(traj_theta)
            )
            if model_data['name']=='dthp':
                traj_theta_dthp = traj_theta
            else: 
                traj_theta_sir = traj_theta
    
            # Final particle filter step for the last time step (forecasting)
            if t == num_timesteps - 1:
                # Initial state for the model
                ini_state = initial_one_state(initial_state_info, num_state_particles)
                current_state = np.array(ini_state['currentStateParticles'])
                theta = np.median(model_data['current_theta'], axis=0)
                theta = untransform_theta(theta, initial_theta_info)
                
                # Run Particle Filter for forecasting
                PF_results = Particle_Filter(model, model_data['name'], model_data['state_names'], current_state, theta, 
                                             model_data['theta_names'], observed_data, num_state_particles, observation_distribution,
                                             resampling_method, forecast_days=forecast_days, add=1, end=True, n_jobs=n_jobs)
                traj_state = PF_results['traj_state']
                if model_data['name']=='dthp':
                    traj_state_dthp = traj_state
                else: 
                    traj_state_sir = traj_state
        
                # Update progress bar
        if show_progress:
            progress_bar.update(1)
    
        # Perform garbage collection to free up memory
        gc.collect()
    
    # Close the progress bar
    if show_progress:
        progress_bar.close()
    
    # Calculate the evidence for both models
    EV_dthp = prod_window(model_evid_dthp, window_size=tw)
    EV_sir = prod_window(model_evid_sir, window_size=tw)
    
    # Calculate the weights for the models
    W_dthp = EV_dthp / (EV_dthp + EV_sir)
    W_sir = 1 - W_dthp
    
    # Extend weights if forecast
    W_dthp = extend_array(W_dthp, num_timesteps + forecast_days)
    
    # Return the results from both models
    return {
        'weight_dthp': W_dthp,
        'weight_sir': W_sir,
        'traj_theta_dthp': traj_theta_dthp,
        'traj_theta_sir': traj_theta_sir,
        'traj_state_dthp': traj_state_dthp,
        'traj_state_sir': traj_state_sir,
        'ESS_theta_dthp': ESS_theta_dthp,
        'ESS_theta_sir': ESS_theta_sir,
    }


def extend_array(arr, target_length):
    if len(arr) < target_length:
        return np.pad(arr, (0, target_length - len(arr)), mode='edge')
    return arr[:target_length]


def prod_window(arr, window_size=1):
    window_size = min (arr.shape[0], window_size)
    cumprod_result = []
    window_prod = 1  
    
    for i, value in enumerate(arr):
        window_prod *= value
        
        cumprod_result.append(window_prod)
        if (i + 1) % window_size == 0:
            window_prod = 1
    
    return np.array(cumprod_result)

    
def Evidence(theta_weights, like):
    """
    Return the evidence at a given time
    """
    return np.average(like, weights = theta_weights)
