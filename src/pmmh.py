"""
Particle Marginal Metropolis-Hastings (PMMH) rejuvenation kernel and the
log-prior (with Jacobian adjustment for transformed parameters).
"""

import numpy as np
from scipy.stats import uniform, norm, truncnorm, lognorm, gamma, invgamma

from src.priors import (
    initial_one_state,
    untransform_theta,
)
from src.particle_filter import Particle_Filter


def PMMH_kernel(
    model, model_type, Z_current, current_theta_particles, current_state_particles, theta_names,
    observed_data, state_names, initial_theta_info, initial_state_info, num_state_particles,
    theta_mean_current, theta_covariance_current, observation_distribution,
    resampling_method, m, t, pmmh_moves, c, N, n_jobs=10
):
    """
    Perform PMMH moves to rejuvenate one theta particle (index `m`).

    Returns a dict with the (possibly updated) marginal log-likelihood 'Z',
    'log_prior_theta', 'state', 'theta', and the move acceptance rate 'acc'.
    """
    acc = 0
    I = 1e-5 * np.eye(theta_covariance_current.shape[0])
    theta_covariance_current = c * theta_covariance_current + I

    state_current = current_state_particles
    theta_current = current_theta_particles[m]
    log_prior_current = log_prior(initial_theta_info, theta_current)

    for _ in range(pmmh_moves):
        if theta_mean_current.shape[0] == 1:
            theta_proposal = np.random.normal(theta_mean_current, np.sqrt(theta_covariance_current[0, 0]))
        else:
            theta_proposal = np.random.multivariate_normal(theta_mean_current, theta_covariance_current)

        log_prior_proposal = log_prior(initial_theta_info, theta_proposal)
        if not np.isfinite(log_prior_proposal):
            continue

        current = Z_current + log_prior_current

        ini_state = initial_one_state(initial_state_info, num_state_particles)
        current_state = np.array(ini_state['currentStateParticles'])
        untrans_theta_proposal = untransform_theta(theta_proposal, initial_theta_info)

        PF_results = Particle_Filter(
            model, model_type, state_names, current_state, untrans_theta_proposal,
            theta_names, observed_data, num_state_particles,
            observation_distribution, resampling_method, N=N, n_jobs=n_jobs
        )

        Z_proposal = PF_results['margLogLike']
        state_proposal = PF_results['particle_state']
        proposal = Z_proposal + log_prior_proposal

        # 1e-12 offsets avoid numerical instability in the log-density comparison
        proposal += log_multivariate_normal_pdf(theta_current, theta_mean_current, theta_covariance_current) + 1e-12
        current += log_multivariate_normal_pdf(theta_proposal, theta_mean_current, theta_covariance_current) + 1e-12

        alpha = np.exp(proposal - current)

        if np.isfinite(alpha) and np.random.uniform() < min(1, alpha):
            Z_current = Z_proposal
            state_current = state_proposal
            theta_current = theta_proposal
            log_prior_current = log_prior_proposal
            acc += 1

    return {
        'Z': Z_current,
        'log_prior_theta': log_prior_current,
        'state': state_current,
        'theta': theta_current,
        'acc': acc / pmmh_moves,
    }


def log_multivariate_normal_pdf(x, mean, cov):
    """Log multivariate-normal density, without the normalising constant."""
    diff = x - mean
    if mean.shape[0] == 1:
        return -0.5 * (diff ** 2 / cov[0, 0])
    cov_inv = np.linalg.inv(cov)
    return -0.5 * np.dot(diff.T, np.dot(cov_inv, diff))


def log_prior(initial_theta_info, theta):
    """
    Log-prior of `theta` (given in transformed space) with the Jacobian
    adjustment for whichever transform (log/logit/none) each parameter uses.
    """
    theta_names = list(initial_theta_info.keys())
    total_log_prior = 0
    jacobian_adjustment = 0

    for i, value in enumerate(theta):
        lower, upper, mean, std, distribution, trans = initial_theta_info[theta_names[i]]['prior']

        if trans == 'log':
            jacobian_adjustment = value
            theta_original = np.exp(value)
        elif trans == 'logit':
            theta_original = 1 / (1 + np.exp(-value))
            jacobian_adjustment = np.log(theta_original) + np.log(1 - theta_original)
        else:
            theta_original = value
            jacobian_adjustment = 0

        if distribution == 'uniform':
            lp = uniform.logpdf(theta_original, loc=lower, scale=upper - lower)
        elif distribution == 'normal':
            lp = norm.logpdf(theta_original, loc=mean, scale=std)
        elif distribution == 'truncnorm':
            a, b = (lower - mean) / std, (upper - mean) / std
            lp = truncnorm.logpdf(theta_original, a, b, loc=mean, scale=std)
        elif distribution == 'lognormal':
            lp = lognorm.logpdf(theta_original, loc=mean, scale=std)
        elif distribution == 'gamma':
            lp = gamma.logpdf(theta_original, lower, scale=upper)
        elif distribution == 'invgamma':
            lp = invgamma.logpdf(theta_original, lower, scale=upper)
        else:
            raise ValueError(f"Unsupported distribution type: {distribution}")

        total_log_prior += lp

    return total_log_prior + jacobian_adjustment
