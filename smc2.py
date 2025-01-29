
# import numpy as np
# import pandas as pd
# import gc
# from joblib import Parallel, delayed
# from tqdm import tqdm
# from ssm_prior_draw import*
# from state_process import state_transition
# from smc import Particle_Filter
# from pmmh import PMMH_kernel
# from resampling import resampling_style
# #from observation_dist import compute_log_weight


# def SMC_squared(
#     model, initial_state_info, initial_theta_info, observed_data, num_state_particles,
#     num_theta_particles, observation_distribution, resampling_threshold=0.5, 
#     pmmh_moves=5, c=0.5, n_jobs=10, resampling_method='stratified', tw=None, 
#     Real_time=False, SMC2_results=None, forecast_days=0, show_progress=True
# ):
#     """
#     Perform (Oline) Sequential Monte Carlo Squared (SMC^2/O-SMC^2) for a given model to estimate
#     the state and parameters

#     Parameters:
#     model (func): The model to be used for particle filtering.
#     initial_state_info (dict): Information for initializing state particles.
#     initial_theta_info (dict): Information for initializing theta particles.
#     observed_data (pd.DataFrame): Observed data to fit the model to.
#     num_state_particles (int): Number of state particles to use in the filter.
#     num_theta_particles (int): Number of theta particles.
#     observation_distribution (func): Type of observation likelihood.
#     resampling_threshold (float): Threshold for resampling based on effective sample size (ESS).
#     pmmh_moves (int): Number of PMMH move in the rejuvenation step.
#     c (int): scaling factor for the covariance matrix in the PMMH kernel.
#     n_jobs (int): Number of processor in the PMMH parallel computing
#     resampling_method (str): Method for resampling ('stratified', etc.).
#     tw (int): Window size for the (O-SMC^2).
#     SMC2_results (dict) : Previous SMC^2 results for previous days that are considered as prior here
#     Real_time (bool) : Whether to use SMC^2 time base on revious SMC^2 results
#     forecast_days (int): Number of forecast days to perform.
#     show_progress (bool): Whether to show a progress bar.

#     Returns:
#     dict: A dictionary with the following keys:
#         - 'margLogLike': Marginal log-likelihood.
#         - 'trajState': State trajectories.
#         - 'trajtheta': Theta trajectories.
#         - 'ESS': Effective sample size over time.
#         - 'acc': Acceptance rate over time.
#         - 'Nx': Number of particles over time.
#     """
#     num_timesteps = len(observed_data)
    
#     # Initialize arrays to store results
#     Z_arr = np.zeros((num_theta_particles, num_timesteps))
#     LogLik = np.zeros(num_theta_particles)
#     likelihood_increment = np.ones(num_theta_particles)

#     # Initialize theta and state particles
#     theta_weights = np.ones(num_theta_particles) / num_theta_particles
#     particle_weights = np.ones((num_theta_particles, num_state_particles)) / num_state_particles
#     ESS_theta_t = np.zeros(num_timesteps)
#     Nx = np.zeros(num_timesteps)
#     acceptance_rate = np.zeros(num_timesteps)

#     # Initialize theta particles
#     initialization_theta = initial_theta(initial_theta_info, num_theta_particles)
#     current_theta_particles = initialization_theta['currentThetaParticles']
#     theta_names = initialization_theta['thetaName']

#     # Initialize state particles
#     initialization_state = initial_state(initial_state_info, num_theta_particles, num_state_particles)
#     current_state_particles_all = initialization_state['currentStateParticles']
#     state_history = np.zeros((num_timesteps, num_theta_particles, num_state_particles, len(initialization_state['stateName'])))
#     state_names = initialization_state['stateName']
   
#     if Real_time:
#         current_theta_particles = SMC2_results['current_theta_particles']
#         current_state_particles_all = SMC2_results['current_state_particles_all']

#     state_history[0] = current_state_particles_all

#     # Max number of theta particles for efficiency
#     Nx_max = 2000
#     num_theta_particles = min(num_theta_particles, Nx_max)

#     if tw is None:
#         tw = num_timesteps

#     # Initialize trajectory storage for theta
#     traj_theta = [{key: [] for key in ['time'] + theta_names} for _ in range(num_theta_particles)]

#     # Initialize progress bar if required
#     if show_progress:
#         progress_bar = tqdm(total=num_timesteps, desc="SMC^2 Progress")

#     # Main loop over time steps
#     for t in range(num_timesteps):
#         t_start, t_end = (0, 0) if t == 0 else (t - 1, t)
#         current_data_point = observed_data.iloc[t]

#         # Function to process each theta particle
#         def process_particle_theta(theta_idx):
#             trans_theta = current_theta_particles[theta_idx]
#             theta = untransform_theta(trans_theta, initial_theta_info)
#             state_particles = current_state_particles_all[theta_idx]
        
#             # Solve model for state particles and theta
#             trajectories = state_transition(model, theta, state_particles, state_names, theta_names, t_start, t_end)
#             model_points = trajectories.to_numpy()

#             # Compute log weights for the model
#             weights = observation_distribution(current_data_point, trajectories, theta, theta_names)

#             # Normalize and resample weights
#             A = np.max(weights)
#             weights_mod = np.ones_like(weights) if A < -1e2 else np.exp(weights - A)
#             normalized_weights = weights_mod / np.sum(weights_mod)
#             resampled_indices = resampling_style(normalized_weights, resampling_method)
#             current_state_particles = model_points[resampled_indices]

#             # Likelihood increment for this particle
#             likelihood_increment_theta = np.mean(np.exp(weights))
#             likelihood_increment_theta = max(likelihood_increment_theta, 1e-12)
        
#             return {
#                 'state_particles': current_state_particles,
#                 'likelihood': likelihood_increment_theta,
#                 'theta': trans_theta,
#             }

#         # Process all theta particles in parallel
#         particles_update_theta = Parallel(n_jobs=n_jobs)(delayed(process_particle_theta)(m) for m in range(num_theta_particles))

#         # Update theta and state particles
#         current_state_particles_all = np.array([p['state_particles'] for p in particles_update_theta])
#         current_theta_particles = np.array([p['theta'] for p in particles_update_theta])
#         likelihood_increment = np.array([p['likelihood'] for p in particles_update_theta])
#         state_history[t] = current_state_particles_all  

#         # Update likelihood and log weights
#         Z_arr[:, t] = np.log(likelihood_increment)
#         Z = np.sum(Z_arr[:, max(0, t - tw):t + 1], axis=1)
#         LogLik = np.sum(Z_arr[:, :], axis=1) - Z

#         if t > 0:
#             theta_weights *= likelihood_increment
#         else:
#             theta_weights = 1 / num_state_particles * np.ones(num_theta_particles)

#         theta_weights /= np.sum(theta_weights)
#         ESS_theta = 1 / (np.sum(theta_weights ** 2))
#         ESS_theta_t[t] = ESS_theta

#         # Resampling-move step (Rejuvenation)
#         if ESS_theta < resampling_threshold * num_theta_particles:
#             # Resampling
#             resampled_indices_theta = resampling_style(theta_weights, resampling_method)
#             theta_mean = np.average(current_theta_particles, axis=0, weights=theta_weights)
#             theta_covariance = np.cov(current_theta_particles.T, ddof=0, aweights=theta_weights)
#             current_theta_particles = current_theta_particles[resampled_indices_theta]
#             current_state_particles_all = current_state_particles_all[resampled_indices_theta]
#             state_history[t] = current_state_particles_all

#             # Reset the weights and Run the PMCMC kernel 
#             theta_weights = np.ones(num_theta_particles) / num_theta_particles
#             new_particles = Parallel(n_jobs=10)(delayed(PMMH_kernel)(
#                 model, Z[m], current_theta_particles, state_history, theta_names,
#                 observed_data.iloc[max(0, t - tw):t + 1], state_names, initial_theta_info, 
#                 num_state_particles, theta_mean, theta_covariance, observation_distribution,
#                 resampling_method, m, t, tw, pmmh_moves, c) for m in range(num_theta_particles))

#             # Update particles and states
#             current_theta_particles = np.array([new['theta'] for new in new_particles])
#             current_state_particles_all = np.array([new['state'] for new in new_particles])
#             acceptance_rate[t] = np.mean([new['acc'] for new in new_particles])
#             state_history[t] = current_state_particles_all
#             Z = np.array([new['Z'] for new in new_particles])

#         LogLik += Z

#         # Update trajectory for each theta particle
#         traj_theta = Parallel(n_jobs=10)(
#             delayed(lambda traj, j: pd.DataFrame(
#                 {'time': list(traj['time']) + [t], 
#                  **{name: list(traj[name]) + [untransform_theta(current_theta_particles[j], initial_theta_info)[i]] 
#                     for i, name in enumerate(theta_names)}}
#             ))(traj, j) 
#             for j, traj in enumerate(traj_theta)
#         )

#         # Final particle filter step for the last time step
#         if t == num_timesteps - 1:
#             if Real_time:
#                 current_state=SMC2_results['current_state_particles']
#             else:
#                 ini_state = initial_one_state(initial_state_info, num_state_particles)
#                 current_state = np.array(ini_state['currentStateParticles'])
#             theta = np.median(current_theta_particles, axis=0)
#             theta = untransform_theta(theta, initial_theta_info)
#             PF_results = Particle_Filter(model, state_names, current_state, theta, theta_names,
#                                          observed_data, num_state_particles, observation_distribution, 
#                                           resampling_method, forecast_days=forecast_days, add=1, end=True, n_jobs=n_jobs)
#             traj_state = PF_results['traj_state']
#             current_state_particles =PF_results['particle_state']

#         # Update progress bar
#         if show_progress:
#             progress_bar.update(1)

#         # Perform garbage collection to free up memory
#         gc.collect()

   
#     non_zero_mask = acceptance_rate != 0

#     filtered_indices = np.arange(num_timesteps)[non_zero_mask]
#     filtered_acc = acceptance_rate[non_zero_mask]
#     # Close progress bar if it was shown
#     if show_progress:
#         progress_bar.close()

#     # Return results
#     return {
#         'margLogLike': np.mean(LogLik),
#         'trajState': traj_state,
#         'trajtheta': traj_theta,
#         'current_theta_particles': current_theta_particles,
#         'current_state_particles': current_state_particles,
#         'current_state_particles_all': current_state_particles_all,
#         'state_history': state_history,
#         'ESS': ESS_theta_t,
#         'acc': filtered_acc
#     }




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


def SMC_squared(
    model, initial_state_info, initial_theta_info, observed_data, num_state_particles,
    num_theta_particles, observation_distribution, resampling_threshold=0.5, 
    pmmh_moves=5, c=0.5, n_jobs=10, resampling_method='stratified', tw=None, 
    Real_time=False, SMC2_results=None, forecast_days=0, show_progress=True
):
    """
    Perform (Oline) Sequential Monte Carlo Squared (SMC^2/O-SMC^2) for a given model to estimate
    the state and parameters

    Parameters:
    model (func): The model to be used for particle filtering.
    initial_state_info (dict): Information for initializing state particles.
    initial_theta_info (dict): Information for initializing theta particles.
    observed_data (pd.DataFrame): Observed data to fit the model to.
    num_state_particles (int): Number of state particles to use in the filter.
    num_theta_particles (int): Number of theta particles.
    observation_distribution (func): Type of observation likelihood.
    resampling_threshold (float): Threshold for resampling based on effective sample size (ESS).
    pmmh_moves (int): Number of PMMH move in the rejuvenation step.
    c (int): scaling factor for the covariance matrix in the PMMH kernel.
    n_jobs (int): Number of processor in the PMMH parallel computing
    resampling_method (str): Method for resampling ('stratified', etc.).
    tw (int): Window size for the (O-SMC^2).
    SMC2_results (dict) : Previous SMC^2 results for previous days that are considered as prior here
    Real_time (bool) : Whether to use SMC^2 time base on revious SMC^2 results
    forecast_days (int): Number of forecast days to perform.
    show_progress (bool): Whether to show a progress bar.

    Returns:
    dict: A dictionary with the following keys:
        - 'log_modelevidence': Evolution of the model evidence over the time.
        - 'trajState': State trajectories.
        - 'trajtheta': Theta trajectories.
        - 'ESS': Effective sample size over time.
        - 'acc': Acceptance rate over time.
        - 'Nx': Number of particles over time.
    """
    num_timesteps = len(observed_data)
    
    # Initialize arrays to store results
    Z_arr = np.zeros((num_theta_particles, num_timesteps))
    likelihood_increment = np.ones(num_theta_particles)
    log_model_evid = np.zeros(num_timesteps)

    # Initialize theta and state particles
    theta_weights = np.ones((num_theta_particles, num_timesteps )) / num_theta_particles
    particle_weights = np.ones((num_theta_particles, num_state_particles)) / num_state_particles
    ESS_theta = np.zeros(num_timesteps)
    Nx = np.zeros(num_timesteps)
    acceptance_rate = np.zeros(num_timesteps)

    # Initialize theta particles
    initialization_theta = initial_theta(initial_theta_info, num_theta_particles)
    current_theta_particles = initialization_theta['currentThetaParticles']
    theta_names = initialization_theta['thetaName']

    # Initialize state particles
    initialization_state = initial_state(initial_state_info, num_theta_particles, num_state_particles)
    current_state_particles_all = initialization_state['currentStateParticles']
    state_history = np.zeros((num_timesteps, num_theta_particles, num_state_particles, len(initialization_state['stateName'])))
    state_names = initialization_state['stateName']
   
    if Real_time:
        current_theta_particles = SMC2_results['current_theta_particles']
        current_state_particles_all = SMC2_results['current_state_particles_all']

    state_history[0] = current_state_particles_all

    # Max number of theta particles for efficiency
    Nx_max = 2000
    num_theta_particles = min(num_theta_particles, Nx_max)

    if tw is None:
        tw = num_timesteps

    # Initialize trajectory storage for theta
    traj_theta = [{key: [] for key in ['time'] + theta_names} for _ in range(num_theta_particles)]

    # Initialize progress bar if required
    if show_progress:
        progress_bar = tqdm(total=num_timesteps, desc="SMC^2 Progress")

    # Main loop over time steps
    for t in range(num_timesteps):
        t_start, t_end = (0, 0) if t == 0 else (t - 1, t)
        current_data_point = observed_data.iloc[t]

        # Function to process each theta particle
        def process_particle_theta(theta_idx):
            trans_theta = current_theta_particles[theta_idx]
            theta = untransform_theta(trans_theta, initial_theta_info)
            state_particles = current_state_particles_all[theta_idx]
        
            # Solve model for state particles and theta
            trajectories = state_transition(model, theta, state_particles, state_names, theta_names, t_start, t_end)
            model_points = trajectories.to_numpy()

            # Compute log weights for the model
            weights = observation_distribution(current_data_point, trajectories, theta, theta_names)

            # Normalize and resample weights
            A = np.max(weights)
            weights_mod = np.ones_like(weights) if A < -1e2 else np.exp(weights - A)
            normalized_weights = weights_mod / np.sum(weights_mod)
            resampled_indices = resampling_style(normalized_weights, resampling_method)
            current_state_particles = model_points[resampled_indices]

            # Likelihood increment for this particle
            likelihood_increment_theta = np.mean(np.exp(weights))
            likelihood_increment_theta = max(likelihood_increment_theta, 1e-12)
        
            return {
                'state_particles': current_state_particles,
                'likelihood': likelihood_increment_theta,
                'theta': trans_theta,
            }

        # Process all theta particles in parallel
        particles_update_theta = Parallel(n_jobs=n_jobs)(delayed(process_particle_theta)(m) for m in range(num_theta_particles))

        # Update theta and state particles
        current_state_particles_all = np.array([p['state_particles'] for p in particles_update_theta])
        current_theta_particles = np.array([p['theta'] for p in particles_update_theta])
        likelihood_increment = np.array([p['likelihood'] for p in particles_update_theta])
        state_history[t] = current_state_particles_all  

        # Store the incremental log  likelihood and log weights
        Z_arr[:, t] = np.log(likelihood_increment)
        Z_w = Z_arr[:, max(0, t - tw):t + 1] # restrist the interest window size

        if t > 0:
            theta_weights[:, t] = theta_weights[:, t-1] * likelihood_increment
            log_model_evid[t] = log_model_evid[t-1] + np.log(Evidence(theta_weights[:, t-1], likelihood_increment))
        else:
            theta_weights[:, t] = theta_weights[:, t] * likelihood_increment
            log_model_evid[t] = np.log(Evidence(theta_weights[:, t], likelihood_increment))

        theta_weights[:, t] /= np.sum(theta_weights[:, t])
        ESS_theta[t] = 1 / (np.sum(theta_weights[:, t] ** 2))
        
        # Resampling-move step (Rejuvenation)
        if ESS_theta[t] < resampling_threshold * num_theta_particles:
            # Resampling and  Reset the weights 
            resampled_indices_theta = resampling_style(theta_weights[:, t], resampling_method)
            Z_w = Z_w[resampled_indices_theta]

            theta_mean = np.average(current_theta_particles, axis=0, weights=theta_weights[:, t])
            theta_covariance = np.cov(current_theta_particles.T, ddof=0, aweights=theta_weights[:, t])
            theta_weights[:, t] = np.ones(num_theta_particles) / num_theta_particles

            current_theta_particles = current_theta_particles[resampled_indices_theta]
            current_state_particles_all = current_state_particles_all[resampled_indices_theta]
            state_history[t] = current_state_particles_all


            # Run the PMCMC kernel 
            new_particles = Parallel(n_jobs=10)(delayed(PMMH_kernel)(
                model, Z_w, current_theta_particles, state_history, theta_names,
                observed_data.iloc[max(0, t - tw):t + 1], state_names, initial_theta_info, 
                num_state_particles, theta_mean, theta_covariance, observation_distribution,
                resampling_method, m, t, tw, pmmh_moves, c) for m in range(num_theta_particles))

            # Update particles and states
            current_theta_particles = np.array([new['theta'] for new in new_particles])
            current_state_particles_all = np.array([new['state'] for new in new_particles])
            acceptance_rate[t] = np.mean([new['acc'] for new in new_particles])
            state_history[t] = current_state_particles_all
            Z_w = np.array([new['Z_w_m'] for new in new_particles])
          
        # Z_arr[:, max(0, t - tw):t + 1] = Z_w # Update the window size of incremental log likelihood
        
            

        # store the trajectory for each theta particle
        traj_theta = Parallel(n_jobs=n_jobs)(
            delayed(lambda traj, j: pd.DataFrame(
                {'time': list(traj['time']) + [t], 
                 **{name: list(traj[name]) + [untransform_theta(current_theta_particles[j], initial_theta_info)[i]] 
                    for i, name in enumerate(theta_names)}}
            ))(traj, j) 
            for j, traj in enumerate(traj_theta)
        )

        # Final particle filter step for the last time step
        if t == num_timesteps - 1:
            if Real_time:
                current_state=SMC2_results['current_state_particles']
            else:
                ini_state = initial_one_state(initial_state_info, num_state_particles)
                current_state = np.array(ini_state['currentStateParticles'])
            theta = np.median(current_theta_particles, axis=0)
            theta = untransform_theta(theta, initial_theta_info)
            PF_results = Particle_Filter(model, state_names, current_state, theta, theta_names,
                                         observed_data, num_state_particles, observation_distribution, 
                                          resampling_method, forecast_days=forecast_days, add=1, end=True, n_jobs=n_jobs)
            traj_state = PF_results['traj_state']
            current_state_particles =PF_results['particle_state']

        # Update progress bar
        if show_progress:
            progress_bar.update(1)

        # Perform garbage collection to free up memory
        gc.collect()

   
    non_zero_mask = acceptance_rate != 0

    filtered_indices = np.arange(num_timesteps)[non_zero_mask]
    filtered_acc = acceptance_rate[non_zero_mask]
    # Close progress bar if it was shown
    if show_progress:
        progress_bar.close()

    # Return results
    return {
        'log_modelevidence': log_model_evid,
        'margiLogLike': log_model_evid[-1],
        'trajState': traj_state,
        'trajtheta': traj_theta,
        'current_theta_particles': current_theta_particles,
        'current_state_particles': current_state_particles,
        'current_state_particles_all': current_state_particles_all,
        'state_history': state_history,
        'ESS': ESS_theta,
        'acc': filtered_acc
    }

    
def Evidence(theta_weights, like):
    """
    Return the evidence at a given time
    """
    return np.average(like, weights = theta_weights)

