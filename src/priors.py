"""
Prior handling for the particle filter / SMC^2 sampler.

Contains:
- transform / untransform of theta parameters (log, logit, or none)
- random draws from the supported prior distributions
- initialisation of state and theta particle clouds
"""

import numpy as np
from scipy.stats import truncnorm, gamma, invgamma


# ---------------------------------------------------------------------------
# Parameter transforms (used so that unconstrained PMMH proposals stay valid)
# ---------------------------------------------------------------------------

def logit(x):
    return np.log(x / (1 - x))


def inv_logit(x):
    return 1 / (1 + np.exp(-x))


def transform_theta(theta, initial_theta_info):
    """Apply the transform (log, logit, or none) declared for each parameter."""
    transformed_theta = np.zeros_like(theta)

    for i, (param, info) in enumerate(initial_theta_info.items()):
        trans = info.get('prior', ['none'])[-1]
        if trans == 'log':
            transformed_theta[i] = np.log(theta[i])
        elif trans == 'logit':
            transformed_theta[i] = logit(theta[i])
        else:
            transformed_theta[i] = theta[i]

    return transformed_theta


def untransform_theta(theta, initial_theta_info):
    """Reverse `transform_theta`, returning parameters to their natural scale."""
    untransformed_theta = np.zeros_like(theta)

    for i, (param, info) in enumerate(initial_theta_info.items()):
        trans = info.get('prior', ['none'])[-1]
        if trans == 'log':
            untransformed_theta[i] = np.exp(theta[i])
        elif trans == 'logit':
            untransformed_theta[i] = inv_logit(theta[i])
        else:
            untransformed_theta[i] = theta[i]

    return untransformed_theta


# ---------------------------------------------------------------------------
# Random draws
# ---------------------------------------------------------------------------

def draw_value(lower, upper, mean, std, distribution, transform=None):
    """
    Draw one random value from a named prior distribution.

    distribution: 'uniform', 'normal', 'lognormal', 'gamma', 'invgamma', 'truncnorm'
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


# ---------------------------------------------------------------------------
# Particle cloud initialisation
# ---------------------------------------------------------------------------

def initial_one_state(state_info, num_state_particles):
    """Initialise a single set of state particles from their priors."""
    state_names = list(state_info.keys())
    current_state_particles = np.zeros((num_state_particles, len(state_names)))

    for i in range(num_state_particles):
        current_state_particles[i] = [
            draw_value(*state_info[state]['prior']) for state in state_names
        ]

    return {
        'currentStateParticles': current_state_particles,
        'stateName': state_names,
    }


def initial_state(state_info, num_theta_particles, num_state_particles):
    """Initialise a state-particle cloud for every theta particle."""
    state_names = list(state_info.keys())
    current_state_particles_all = np.zeros(
        (num_theta_particles, num_state_particles, len(state_names))
    )

    for j in range(num_theta_particles):
        for i in range(num_state_particles):
            current_state_particles_all[j, i, :] = [
                draw_value(*state_info[state]['prior']) for state in state_names
            ]

    return {
        'currentStateParticles': current_state_particles_all,
        'stateName': state_names,
    }


def initial_theta(initial_theta_info, num_theta_particles):
    """Initialise (and transform) the theta particle cloud from the priors."""
    theta_names = list(initial_theta_info.keys())
    current_theta_particles = np.zeros((num_theta_particles, len(theta_names)))

    for i in range(num_theta_particles):
        theta_values = [
            draw_value(*initial_theta_info[param]['prior']) for param in theta_names
        ]
        current_theta_particles[i] = transform_theta(theta_values, initial_theta_info)

    return {
        'currentThetaParticles': current_theta_particles,
        'thetaName': theta_names,
    }
