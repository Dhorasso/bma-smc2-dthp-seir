################################################################################################################
# This file contains the different observation distribution
#  **IMPORTANT NOTE**: (1) Make sure make sure the column name of the obseration is 'obs' or  change 'obs' here by the actual column name
#                      (2) Make sure make sure the column name of the model state you use to link with observastio is
#                          'NI' or change it here to have the same name
##################################################################################################################


import numpy as np
from scipy.stats import poisson, norm, nbinom

# Poisson log-likelihood
def obs_dist_poisson(observed_data, model_data, theta, theta_names):
    epsi = 1e-4
    model_est_case = np.maximum(epsi, model_data['NI'].to_numpy())
    log_likelihoods = poisson.logpmf(observed_data['obs'], mu=model_est_case)
    log_likelihoods[np.isnan(log_likelihoods) | np.isinf(log_likelihoods)] = -np.inf
    return log_likelihoods

# Normal log-likelihood
def obs_dist_normal(observed_data, model_data, theta, theta_names):
    epsi = 1e-4
    model_est_case = np.maximum(epsi, model_data['NI'].to_numpy())
    param = dict(zip(theta_names, theta))
    sigma_normal = param.get('phi', 0.1)  # Default value for 'phi' if not provided
    log_likelihoods = norm.logpdf(np.log(epsi + observed_data['obs']),
                                  loc=np.log(epsi + model_est_case),
                                  scale=sigma_normal)
    log_likelihoods[np.isnan(log_likelihoods) | np.isinf(log_likelihoods)] = -np.inf
    return log_likelihoods

# Normal approximation to Negative Binomial log-likelihood
def obs_dist_normal_approx_NB(observed_data, model_data, theta, theta_names):
    epsi = 1e-4
    model_est_case = np.maximum(epsi, model_data['NI'].to_numpy())
    param = dict(zip(theta_names, theta))
    overdispersion = param.get('phi', 0.1)  # Default value for 'phi' if not provided
    variance = model_est_case * (1 + overdispersion * model_est_case)
    variance = np.maximum(variance, 1)  # Ensure variance is at least 1
    log_likelihoods = norm.logpdf(observed_data['obs'], loc=model_est_case, scale=np.sqrt(variance))
    log_likelihoods[np.isnan(log_likelihoods) | np.isinf(log_likelihoods)] = -np.inf
    return log_likelihoods

# Negative Binomial log-likelihood
def obs_dist_negative_binomial(observed_data, model_data, theta, theta_names):
    epsi = 1e-4
    model_est_case = np.maximum(epsi, model_data['NI'].to_numpy())
    param = dict(zip(theta_names, theta))
    overdispersion = param.get('phi', 0.1)  # Default value for 'phi' if not provided
    log_likelihoods = nbinom.logpmf(observed_data['obs'], 1 / overdispersion, 1 / (1 + overdispersion * model_est_case))
    log_likelihoods[np.isnan(log_likelihoods) | np.isinf(log_likelihoods)] = -np.inf
    return log_likelihoods
