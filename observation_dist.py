################################################################################################################
# This script contains the different observation distribution
#  **IMPORTANT NOTE**: (1) Make sure make sure the column name of the obseration is 'obs' or  change 'obs' here by the actual column name
#                      (2) Make sure make sure the column name of the model state you use to link with observastio is
#                          'NI' or change it here to have the same name
##################################################################################################################
import numpy as np
from scipy.stats import poisson, norm, nbinom

# ===============================================================
#  Poisson
# ===============================================================
def obs_dist_poisson(observed_data, model_data, theta, theta_names, pred=False):
    epsi = 1e-4
    model_est_case = np.maximum(epsi, model_data['NI'].to_numpy())

    if pred:
        pred_vals = np.random.poisson(model_est_case)
        model_new = model_data.copy()
        model_new['NI'] = pred_vals
        return model_new

    log_likelihoods = poisson.logpmf(observed_data['obs'], mu=model_est_case)
    log_likelihoods[np.isnan(log_likelihoods) | np.isinf(log_likelihoods)] = -np.inf
    return log_likelihoods


# ===============================================================
#  Normal (log-normal observation)
# ===============================================================
def obs_dist_normal(observed_data, model_data, theta, theta_names, pred=False):
    epsi = 1e-4
    model_est_case = np.maximum(epsi, model_data['NI'].to_numpy())
    param = dict(zip(theta_names, theta))
    sigma_normal = param.get('phi', 0.1)

    if pred:
        mu = np.log(epsi + model_est_case)
        pred_vals = np.random.lognormal(mean=mu, sigma=sigma_normal)
        model_new = model_data.copy()
        model_new['NI'] = pred_vals
        return model_new

    log_likelihoods = norm.logpdf(
        np.log(epsi + observed_data['obs']),
        loc=np.log(epsi + model_est_case),
        scale=sigma_normal
    )
    log_likelihoods[np.isnan(log_likelihoods) | np.isinf(log_likelihoods)] = -np.inf
    return log_likelihoods


# ===============================================================
#  Normal Approximation to NB
# ===============================================================
def obs_dist_normal_approx_NB(observed_data, model_data, theta, theta_names, pred=False):
    epsi = 1e-4
    model_est_case = np.maximum(epsi, model_data['NI'].to_numpy())
    param = dict(zip(theta_names, theta))
    overdispersion = param.get('phi', 0.1)

    variance = model_est_case * (1 + overdispersion * model_est_case)
    variance = np.maximum(variance, 1)
    sd = np.sqrt(variance)

    if pred:
        pred_vals = np.random.normal(loc=model_est_case, scale=sd)
        model_new = model_data.copy()
        model_new['NI'] = pred_vals
        return model_new

    log_likelihoods = norm.logpdf(observed_data['obs'], loc=model_est_case, scale=sd)
    log_likelihoods[np.isnan(log_likelihoods) | np.isinf(log_likelihoods)] = -np.inf
    return log_likelihoods


# ===============================================================
#  Negative Binomial
# ===============================================================
def obs_dist_negative_binomial(observed_data, model_data, theta, theta_names, pred=False):
    epsi = 1e-4
    model_est_case = np.maximum(epsi, model_data['NI'].to_numpy())
    param = dict(zip(theta_names, theta))
    overdispersion = param.get('phi', 0.1)

    r = 1 / overdispersion
    p = 1 / (1 + overdispersion * model_est_case)

    if pred:
        pred_vals = np.random.negative_binomial(n=r, p=p)
        model_new = model_data.copy()
        model_new['NI'] = pred_vals
        return model_new

    log_likelihoods = nbinom.logpmf(observed_data['obs'], r, p)
    log_likelihoods[np.isnan(log_likelihoods) | np.isinf(log_likelihoods)] = -np.inf
    return log_likelihoods

