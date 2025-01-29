########################################################################################
# This file contains code to handle constraint prameters and draw initail state and 
# parameter particles
##########################################################################################


import numpy as np
from scipy.stats import norm, lognorm, truncnorm, gamma, invgamma


#################################################################################
###### Function to transform/ untransform constrain parametres #####################

# Define the logit and inverse logit functions
def logit(x):
    return np.log(x / (1 - x))

def inv_logit(x):
    return 1 / (1 + np.exp(-x))

def transform_theta(theta, initial_theta_info):
    """
    Apply transformations (log, logit, or none) to theta parameters.

    Parameters:
    - theta: Array of parameter values.
    - initial_theta_info: Dictionary containing prior information and transformation details.

    Returns:
    - transformed_theta: Array of transformed parameter values.
    """
    transformed_theta = np.zeros_like(theta)

    for i, (param, info) in enumerate(initial_theta_info.items()):
        trans = info.get('prior', ['none'])[-1]  # Get the transformation type ('log', 'logit', 'none')
        if trans == 'log':
            transformed_theta[i] = np.log(theta[i])
        elif trans == 'logit':
            transformed_theta[i] = logit(theta[i])
        else:
            transformed_theta[i] = theta[i]  # No transformation

    return transformed_theta

def untransform_theta(theta, initial_theta_info):
    """
    Reverse transformations (log, logit, or none) to return theta parameters to their original scale.

    Parameters:
    - theta: Array of transformed parameter values.
    - initial_theta_info: Dictionary containing prior information and transformation details.

    Returns:
    - untransformed_theta: Array of untransformed parameter values.
    """
    untransformed_theta = np.zeros_like(theta)

    for i, (param, info) in enumerate(initial_theta_info.items()):
        trans = info.get('prior', ['none'])[-1]  # Get the transformation type ('log', 'logit', 'none')
        if trans == 'log':
            untransformed_theta[i] = np.exp(theta[i])
        elif trans == 'logit':
            untransformed_theta[i] = inv_logit(theta[i])
        else:
            untransformed_theta[i] = theta[i]  # No transformation

    return untransformed_theta



def draw_value(lower, upper, mean, std, distribution, transform=None):
    """
    Draw a random value from a specified distribution.

    Parameters:
    - lower (float): Lower bound for uniform, gamma, and invgamma distributions.
    - upper (float): Upper bound for uniform, gamma, and invgamma distributions.
    - mean (float): Mean for normal and lognormal distributions.
    - std (float): Standard deviation for normal and lognormal distributions.
    - distribution (str): Type of distribution ('uniform', 'normal', 'lognormal', 'gamma', 'invgamma', 'truncnorm').
    - transform (function, optional): Transformation function to apply to the drawn value (default is None).

    Returns:
    - value (float): Drawn value from the specified distribution.
    """
    if distribution == 'uniform':
        return np.random.uniform(lower, upper)
    elif distribution == 'normal':
        return np.random.normal(mean, std)
    elif distribution == 'lognormal':
        return np.random.lognormal(mean, std)
    elif distribution == 'gamma':
        return gamma.rvs(lower, scale=upper)
    elif distribution == 'invgamma':
        return invgamma.rvs(lower, scale=upper)
    elif distribution == 'truncnorm':
        a, b = (lower - mean) / std, (upper - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std)
    else:
        raise ValueError("Invalid distribution type")



#################################################################################
###### Functions to draw initial state an prameter particles #####################

        
def initial_one_state(state_info, num_state_particles):
    """
    Initialize state particles for the particle filter.

    Parameters:
    - state_info (dict): Dictionary containing information about initial state distributions.
    - num_state_particles (int): Number of state particles to initialize.

    Returns:
    - result_dict (dict): Dictionary containing initialized state particles and state names.
    """
    state_names = list(state_info.keys())
    current_state_particles = np.zeros((num_state_particles, len(state_names)))
    
    # Generate state particles based on their prior distributions
    for i in range(num_state_particles):
        state_values = [draw_value(*state_info[state]['prior']) for state in state_names]
        current_state_particles[i] = state_values
   
    result_dict = {
        'currentStateParticles': current_state_particles,
        'stateName': state_names,
    }

    return result_dict




def initial_state(state_info, num_theta_particles, num_state_particles):
    """
    Initialize state particles for each theta particle in the particle filter.

    Parameters:
    - state_info (dict): Dictionary containing information about initial state distributions.
    - num_theta_particles (int): Number of theta particles to initialize.
    - num_state_particles (int): Number of state particles to initialize for each theta.

    Returns:
    - result_dict (dict): Dictionary containing initialized state particles and state names for each theta.
    """
    state_names = list(state_info.keys())
    current_state_particles_all = np.zeros((num_theta_particles, num_state_particles, len(state_names)))

    # Generate state particles for each theta particle
    for j in range(num_theta_particles):
        for i in range(num_state_particles):
            state_values = [draw_value(*state_info[state]['prior']) for state in state_names]
            current_state_particles_all[j, i, :] = state_values

    result_dict = {
        'currentStateParticles': current_state_particles_all,
        'stateName': state_names,
    }

    return result_dict


def initial_theta(initial_theta_info, num_theta_particles):
    """
    Initialize theta particles for the particle filter.

    Parameters:
    - initial_theta_info (dict): Dictionary containing information about initial theta distributions.
    - num_theta_particles (int): Number of theta particles to initialize.

    Returns:
    - result_dict (dict): Dictionary containing initialized theta particles and theta names.
    """
    theta_names = list(initial_theta_info.keys())
    current_theta_particles = np.zeros((num_theta_particles, len(theta_names)))

    # Generate theta particles based on their prior distributions
    for i in range(num_theta_particles):
        theta_values = [draw_value(*initial_theta_info[param]['prior']) for param in theta_names]
        theta_values = transform_theta(theta_values, initial_theta_info)
        current_theta_particles[i] = theta_values

    result_dict = {
        'currentThetaParticles': current_theta_particles,
        'thetaName': theta_names,
    }

    return result_dict
