#####################################################################################
# Application of     BMA-SMC^2 for the Scenario C in the paper
# Note: All functions must be in the same folder.
#####################################################################################
# import the necessary libraies

# Standard Libraries
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
from scipy.stats import poisson, norm, nbinom
from numpy.random import binomial, normal, negative_binomial
from joblib import Parallel, delayed  # For parallel computing
from tqdm import tqdm
from scipy.stats import nbinom

# SMC2 Libraries
from models import stochastic_seir_model, dthp_model
from smc_squared import BMA_SMC2
from smc_visualization import trace_smc, plot_smc, compute_model_average
# Style Configuration


############  SEPTP 1:Import/create your dataset ###########################
#### Generate the simulated data with time varying beta ###############
#######################################################################

## True parameter
true_theta = [0.2]  # omega (decay parameter in the geometric kernel)

simulated_data = pd.read_csv('data_SB.csv')
#visulization
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Left panel: observed cases
axes[0].plot(simulated_data['obs'], marker='o')
axes[0].set_title('Observed Cases (obs)')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Cases')
axes[0].grid(True)

# Right panel: reproduction number
axes[1].plot(simulated_data['Rt'])
axes[1].set_title('Reproduction Number (Rt)')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Rt')
axes[1].grid(True)

plt.tight_layout()
plt.show()

#################################################################################################
############ SEPTP 2: Define your observation distribution ######################################
# The observation_dist.py contains some examples, you extend to incorporate multiple dataste
#################################################################################################

# Negative Binomial log-likelihood
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
#################################################################################################
############ SEPTP 3: Run the SMC^2 #####################################################################
# You need to defined initial conditions for the state and prior for the parameter you want to estimate
##########################################################################################################

np.random.seed(123) # Set a seed for reproducibility

# # ##### # setting state and parameter

### SEIR initial state and prior distribution
state_info_seir = {
    'S': {'prior': [N_pop-20, N_pop, 0, 0, 'uniform']},  # Susceptibles
    'E': {'prior': [0, 5, 0, 0, 'uniform']},  # Exposed
    'I': {'prior': [0, 15, 0, 0, 'uniform']},  # Infected
    'R': {'prior': [0, 0, 0, 0, 'uniform']},  # Removed (Recovered or Deceased)
    'NI': {'prior': [0, 0, 0,0, 'uniform']},  # Newly Infected
    'B': {'prior': [0, np.inf, 0.725, 0.01, 'normal']},  # Transmission rate (Beta)
}

theta_info_seir = {
    'sigma': {'prior': [0.2, 0.7, 0.5, 0.05,'truncnorm','log']},  # Latency rate (Inverse of latency period)
    'gamma': {'prior': [0.1, 0.2, 0.16, 0.05,'truncnorm','log']},  # Removal rate (Inverse of infectious period)
    'nu_beta': {'prior': [0.05, 0.2, 0.1, 0.01, 'truncnorm', 'log']},  # Standard deviation of RW process (Beta variability)
    'phi': {'prior': [1e-5, 0.1,0,0, 'uniform','log']}  # Overdispersion parameter
}

# DTHP model state and parameter information
state_info_dthp = {
    'NI': {'prior': [0, 3, 0,0, 'uniform']},  # Newly Infected (lamabda_H(0))
    'Rt': {'prior': [0, np.inf, 4.25, 0.15, 'normal']}  # Reproduction Number
}

theta_info_dthp = {
    'omega_I': {'prior': [0, 1, 0.1, 0.05, 'truncnorm', 'log']},  # Decay parameter in the triggering kernel
    'nu_beta': {'prior': [0.05, 0.2, 0.1, 0.011, 'truncnorm', 'log']},  # Standard deviation of RW process (Beta variability)
    'phi': {'prior': [1e-5, 0.1,0,0, 'uniform','log']}  # Overdispersion parameter
}






fday=14
days=len(simulated_data)-fday

np.random.seed(123) # Set a seed for reproducibility
smc2_results = BMA_SMC2(
    model_seir=stochastic_seir_model,
    model_dthp= dthp_model, 
    initial_state_info_dthp=state_info_dthp , 
    initial_theta_info_dthp=theta_info_dthp , 
    initial_state_info_seir=state_info_seir , 
    initial_theta_info_seir=theta_info_seir , 
    observed_data=simulated_data[:days],
    num_state_particles=200,
    num_theta_particles=400,
    observation_distribution=obs_dist_negative_binomial,
    forecast_days=fday,
    N=N_pop
)



##########################################################################################################
########## SETP4: Visualize the Results ######Test_smc2##############################################################
# You can plot the filtered estimate of the state and parametersx
############################################################################################################

plt.style.use('seaborn-v0_8-white')
separation_point = simulated_data['time'].iloc[-1] - fday

# Evolution of the model weights
w_dthp = smc2_results['weight_dthp']  
w_seir = smc2_results['weight_seir']

# Plotting model weights
fig, ax = plt.subplots(figsize=(11, 4))  
ax.plot(w_dthp, label='Weight DTHP', color='orange', linewidth=2, alpha=0.8)
ax.plot(w_seir, label='Weight SEIR', color='dodgerblue', linewidth=2, alpha=0.8)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_xlabel('Days', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel('Model weights', fontsize=18, fontweight='bold', labelpad=10)
ax.axvline(x=separation_point, color='black', linestyle='--', linewidth=2)
ax.set_title('Scenario C', fontsize=18, fontweight='bold', pad=15)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.legend(fontsize=12, loc='upper right', frameon=True, shadow=True, facecolor='white')
fig.tight_layout()
plt.show()

###################################################################
# state trajectory particles and extract corresponding matrix



# Extract trajectories SEIR model
trajParticles_state_seir = smc2_results['traj_state_seir']
matrix_dict_state_seir = trace_smc(trajParticles_state_seir)
trajParticles_theta_seir = smc2_results['traj_theta_seir']
matrix_dict_theta_seir = trace_smc(trajParticles_theta_seir)
gamma=np.mean(matrix_dict_theta_seir['gamma'][:,-1])
# Calculate the  reproduction number Rt and add it to the satate dict
matrix_dict_state_seir['Rt']=matrix_dict_state_seir['B']/gamma

# Extract trajectories DTHP
trajParticles_state_dthp = smc2_results['traj_state_dthp']
matrix_dict_state_dthp = trace_smc(trajParticles_state_dthp)
trajParticles_theta_dthp = smc2_results['traj_theta_dthp']
matrix_dict_theta_dthp = trace_smc(trajParticles_theta_dthp)

# Compute trajectories model-averging based on the model weights
matrix_dict_state_avg = compute_model_average(matrix_dict_state_dthp, matrix_dict_state_seir, w_dthp, w_seir)


fig, axs = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey='row')  # 2 rows (NI, Rt), 2 cols (DTHP+SEIR, MA)

# --- Add hatched forecast region to all axes ---
for ax in axs.flat:
    ax.axvspan(
        forecast_start,
        forecast_end,
        facecolor='lightgray',
        edgecolor='gray',
        alpha=0.25,
        linewidth=0.0,
        zorder=1
    )

models = ['SEIR','DTHP',  'MA']
colors = [ 'dodgerblue', 'orange', 'green']
matrices = [
    (matrix_dict_state_seir, 'SEIR', colors[0]),
    (matrix_dict_state_dthp, 'DTHP', colors[1]),
    (matrix_dict_state_avg, 'MA', colors[2])
]

# Rolling average of observed data
rolling_obs = simulated_data['obs'].rolling(window=window, min_periods=1).mean()
time = simulated_data['time']
before_sep = time < separation_point
after_sep = time >= separation_point

# --- Incidence row ---
# Col 0: DTHP + SEIR
for matrix_dict, label, color in matrices[:2]:
    ax_inc = axs[0, 0]
    plot_smc(matrix_dict['NI'], ax=ax_inc, separation_point=separation_point, window=window, color=color, label=label)

ax_inc.scatter(time[before_sep], rolling_obs[before_sep], facecolor='black', marker='*', s=60, label='Observed Data (fit)', zorder=2)
ax_inc.scatter(time[after_sep], rolling_obs[after_sep], facecolor='brown', marker='*', s=60, label='Observed Data (forecast)', zorder=2)
ax_inc.axvline(x=separation_point, color='black', linestyle='--', linewidth=2)
# ax_inc.set_title("Incidence: DTHP + SEIR", fontsize=16, fontweight='bold')
ax_inc.set_ylabel("Incidence", fontsize=18, fontweight='bold')
ax_inc.legend(fontsize=14)
ax_inc.set_ylim([-5,160])
ax_inc.tick_params(axis='both', which='major', labelsize=12)

# Col 1: MA
ma_dict, ma_label, ma_color = matrices[2]
ax_inc_ma = axs[0, 1]
plot_smc(ma_dict['NI'], ax=ax_inc_ma, separation_point=separation_point, window=window, color=ma_color, label=ma_label)
ax_inc_ma.scatter(time[before_sep], rolling_obs[before_sep], facecolor='black', marker='*', s=60, label='Observed Data (fit)', zorder=2)
ax_inc_ma.scatter(time[after_sep], rolling_obs[after_sep], facecolor='brown', marker='*', s=60, label='Observed Data (forecast)', zorder=2)
ax_inc_ma.axvline(x=separation_point, color='black', linestyle='--', linewidth=2)
# ax_inc_ma.set_title("Incidence: MA", fontsize=16, fontweight='bold')
ax_inc_ma.tick_params(axis='both', which='major', labelsize=12)
ax_inc_ma.legend(fontsize=14)

# --- Rt row ---
# Col 0: DTHP + SEIR
for matrix_dict, label, color in matrices[:2]:
    ax_rt = axs[1, 0]
    plot_smc(matrix_dict['Rt'], ax=ax_rt, separation_point=separation_point, window=window, color=color, label=label)

ax_rt.plot(time, simulated_data['Rt'].rolling(window=window, min_periods=1).mean(),
           color='brown', lw=3, label='True $R_t$', zorder=3)
ax_rt.axhline(y=1, color='k', linestyle='--', linewidth=3)
# ax_rt.set_title(r"$R_t$: DTHP + SEIR", fontsize=16, fontweight='bold')
ax_rt.set_ylabel(r"Reproduction number $R_t$", fontsize=18, fontweight='bold')
ax_rt.set_xlabel("Days", fontsize=16, fontweight='bold')
ax_rt.legend(fontsize=14, loc='upper center')
ax_rt.tick_params(axis='both', which='major', labelsize=12)

# Col 1: MA
ax_rt_ma = axs[1, 1]
plot_smc(ma_dict['Rt'], ax=ax_rt_ma, separation_point=separation_point, window=window, color=ma_color, label=ma_label)
ax_rt_ma.plot(time, simulated_data['Rt'].rolling(window=window, min_periods=1).mean(),
              color='brown', lw=3, label='True $R_t$', zorder=3)
ax_rt_ma.axhline(y=1, color='k', linestyle='--', linewidth=3)
# ax_rt_ma.set_title(r"$R_t$: MA", fontsize=16, fontweight='bold')
ax_rt_ma.set_xlabel("Days", fontsize=16, fontweight='bold')
ax_rt_ma.set_ylim([0.2, 3.5])
ax_rt_ma.tick_params(axis='both', which='major', labelsize=12)
ax_rt_ma.legend(fontsize=14, loc='upper center')

# Final formatting
for ax in axs.flat:
    ax.grid(True, linestyle='--', alpha=0.9)

plt.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.show()


#############################################################################
# Plot for the parameter trajectories
##############################################################################



L1 = [r'$\omega$', r'$\nu_{1}$', r'$\phi_1$']
L2 = [r'$\sigma$', r'$\gamma$', r'$\nu_{2}$', r'$\phi_2$']
n_plots = len(L1) + len(L2)
nrows, ncols = 3, 3
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10))
axes = axes.flatten()

# Plot DTHP parameters
for i, (state, matrix) in enumerate(matrix_dict_theta_dthp.items()):
    ax = axes[i]
    plot_smc(matrix, ax=ax, show_50ci=True)
    ci_025 = np.percentile(matrix[:, -1], 2.5)
    ci_975 = np.percentile(matrix[:, -1], 97.5)
    median_estimate = np.mean(matrix[:, -1])
    ax.set_title(f'{L1[i]} = {median_estimate:.3f} (95% CrI: [{ci_025:.3f}, {ci_975:.3f}])', 
                 fontsize=18, fontweight='bold')
    ax.set_ylabel(L1[i], fontsize=25, fontweight='bold')

# Plot SEIR parameters
for i, (state, matrix) in enumerate(matrix_dict_theta_seir.items()):
    ax = axes[i + len(L1)]
    plot_smc(matrix, ax=ax, show_50ci=True)
    ci_025 = np.percentile(matrix[:, -1], 2.5)
    ci_975 = np.percentile(matrix[:, -1], 97.5)
    median_estimate = np.mean(matrix[:, -1])
    ax.set_title(f'{L2[i]} = {median_estimate:.3f} (95% CrI: [{ci_025:.3f}, {ci_975:.3f}])', 
                 fontsize=18, fontweight='bold')
    if i <= 1:
        ax.axhline(true_theta[i], color='orange', linestyle='--', linewidth=3, label='True Value')
    ax.set_xlabel('Days', fontsize=16, fontweight='bold')
    ax.set_ylabel(L2[i], fontsize=25, fontweight='bold')

# Hide any unused subplots
for j in range(n_plots, len(axes)):
    fig.delaxes(axes[j])

fig.text(0.02, 0.97, 'C', fontsize=50, fontweight='bold', ha='center', va='center')
for ax in axes[:n_plots]:
    ax.grid(True, linestyle='--', alpha=0.9)

plt.tight_layout()
plt.show()



