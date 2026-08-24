"""
Observation distributions linking the latent 'NI' (new infections) state to the
observed case counts ('obs').

IMPORTANT:
- Each function expects the observation column to be named 'obs' and the model
  state used for the likelihood to be named 'NI'. If your dataset/model uses
  different names, either rename the columns beforehand or adjust the
  hard-coded 'obs' / 'NI' references below.
- Every function follows the same signature so it can be passed as the
  `observation_distribution` argument to `bma_smc2.BMA_SMC2` /
  `particle_filter.Particle_Filter`:
      f(observed_data, model_data, theta, theta_names, pred=False)
  When `pred=True`, the function draws simulated observations instead of
  returning a log-likelihood (used when forecasting beyond the data).
"""

import numpy as np
from scipy.stats import poisson, norm, nbinom


def obs_dist_poisson(observed_data, model_data, theta, theta_names, pred=False):
    epsi = 1e-4
    model_est_case = np.maximum(epsi, model_data['NI'].to_numpy())

    if pred:
        model_new = model_data.copy()
        model_new['NI'] = np.random.poisson(model_est_case)
        return model_new

    log_likelihoods = poisson.logpmf(observed_data['obs'], mu=model_est_case)
    log_likelihoods[np.isnan(log_likelihoods) | np.isinf(log_likelihoods)] = -np.inf
    return log_likelihoods


def obs_dist_normal(observed_data, model_data, theta, theta_names, pred=False):
    """Log-normal observation noise on new infections."""
    epsi = 0.1
    model_est_case = np.maximum(epsi, model_data['NI'].to_numpy())
    sigma_normal = dict(zip(theta_names, theta)).get('phi', 0.1)

    if pred:
        mu = np.log(epsi + model_est_case)
        model_new = model_data.copy()
        model_new['NI'] = np.random.lognormal(mean=mu, sigma=sigma_normal)
        return model_new

    log_likelihoods = norm.logpdf(
        np.log(epsi + observed_data['obs']),
        loc=np.log(epsi + model_est_case),
        scale=sigma_normal,
    )
    log_likelihoods[np.isnan(log_likelihoods) | np.isinf(log_likelihoods)] = -np.inf
    return log_likelihoods


def obs_dist_normal_approx_nb(observed_data, model_data, theta, theta_names, pred=False):
    """Normal approximation to a negative-binomial observation model."""
    epsi = 0.1
    model_est_case = np.maximum(epsi, model_data['NI'].to_numpy())
    overdispersion = dict(zip(theta_names, theta)).get('phi', 0.1)

    variance = np.maximum(model_est_case * (1 + overdispersion * model_est_case), 1)
    sd = np.sqrt(variance)

    if pred:
        model_new = model_data.copy()
        model_new['NI'] = np.random.normal(loc=model_est_case, scale=sd)
        return model_new

    log_likelihoods = norm.logpdf(observed_data['obs'], loc=model_est_case, scale=sd)
    log_likelihoods[np.isnan(log_likelihoods) | np.isinf(log_likelihoods)] = -np.inf
    return log_likelihoods


def obs_dist_negative_binomial(observed_data, model_data, theta, theta_names, pred=False):
    epsi = 0.1
    model_est_case = np.maximum(epsi, model_data['NI'].to_numpy())
    overdispersion = dict(zip(theta_names, theta)).get('phi', 0.1)

    r = 1 / overdispersion
    p = 1 / (1 + overdispersion * model_est_case)

    if pred:
        model_new = model_data.copy()
        model_new['NI'] = np.random.negative_binomial(n=r, p=p)
        return model_new

    log_likelihoods = nbinom.logpmf(observed_data['obs'], r, p)
    log_likelihoods[np.isnan(log_likelihoods) | np.isinf(log_likelihoods)] = -np.inf
    return log_likelihoods
