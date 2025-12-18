#####################################################################################
# Application of     BMA-SMC^2 for the Irish Flu epidemic in the paper
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
from numpy.random import binomial, normal
from joblib import Parallel, delayed  # For parallel computing
from tqdm import tqdm
from matplotlib.ticker import AutoMinorLocator
from matplotlib.dates import MonthLocator, DateFormatter, DayLocator


# SMC2 Libraries
from models import stochastic_seirs_model, dthp_model
from smc_squared import BMA_SMC2
from smc_visualization import trace_smc, plot_smc, compute_model_average
# Style Configuration
# plt.style.use('seaborn-v0_8-white')

############  SEPTP 1:Import your dataset ###########################
# Assuming the uploaded file is named "COVID-19_HPSC_Detailed_Statistics_Profile.csv" OR Download
# from https://COVID19ireland-geohive.hub.arcgis.com/
#######################################################################


file_path = r"COVID-19_HPSC_Detailed_Statistics_Profile.csv"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)
# Restrict the observations to 280 days
st=0
days = st+293
data = df.iloc[st:days].copy()

# Ensure the 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Plotting with matplotlib
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the Confirmed Covid Cases
data['obs']= data['ConfirmedCovidCases'] 
ax.plot(data['Date'], data['ConfirmedCovidCases'], color='blue')

# Formatting the plot
# ax.set_title("COVID-19 Confirmed Cases Over Time", fontsize=16)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Number of Cases", fontsize=12)
ax.grid(True)

# Rotate the x-axis labels for better visibility
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
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

### SEIRS initial state and prior distribution
N_pop = 5.16e6  # Total population
E_0 = 5
I0 =15# Initial number of infected individuals

state_info_seirs = {
    'S': {'prior': [N_pop - E_0-I0, N_pop, 0, 0, 'uniform']},  # Susceptibles
     'E': {'prior': [E_0, E_0, 0, 0, 'uniform']},         # Exposed
    'I': {'prior': [0, I0, 0, 0, 'uniform']},  # Infected
    'R': {'prior': [0, 0, 0, 0, 'uniform']},  # Recovered
    'NI': {'prior': [0, 0, 0, 0, 'uniform']},  # Newly Infected
    'B': {'prior': [0, 1, 0.5, 0.05, 'truncnorm']},  # Transmission rate (Beta)
}

theta_info_seirs = {
    'sigma': {'prior': [1/5, 1/3, 1/4, 0.1, 'truncnorm', 'log']},  # latency rate (inverse of incubation period)
    'gamma': {'prior': [1/7.5, 1/4.5, 1/6, 0.2, 'truncnorm', 'log']}, # removal rate (inverse of infectious period)
    'nu_beta': {'prior': [0.05, 0.15, 0.1, 0.05, 'uniform', 'log']},   # standard deviation RW process 
   'nu_beta': {'prior': [0.05, 0.15, 0.1, 0.02, 'truncnorm', 'log']},   # Volatility in transmission rate
    'phi': {'prior': [1e-5, 0.2, 0, 0, 'uniform', 'log']},  # Overdispersion parameter
}
### DTHP model state and prior distribution
state_info_dthp = {
    'NI': {'prior': [0, I0, 0, 0, 'uniform']},  # Newly Infected (lambda_H(0))
    # 'C_I': {'prior': [0, 0, 0, 0, 'uniform']},  # Cumulative Infected
    'Rt': {'prior': [0, np.inf, 3.2, 0.05, 'normal']},  # Reproduction Number
}

theta_info_dthp = {
    'omega_I': {'prior': [0, 1, 0, 0, 'uniform', 'log']},  # Decay parameter in the triggering kernel
   'nu_beta': {'prior': [0.05, 0.15, 0.1, 0.02, 'truncnorm', 'log']},   # Volatility in transmission rate
    'phi': {'prior': [1e-5, 0.2, 0, 0, 'uniform', 'log']},  # Overdispersion parameter
}



fday = 21
days = len(data)-fday

np.random.seed(123) # Set a seed for reproducibility

smc2_results = BMA_SMC2(
    model_seir=stochastic_seirs_model,
    model_dthp= dthp_model, 
    initial_state_info_dthp=state_info_dthp , 
    initial_theta_info_dthp=theta_info_dthp , 
    initial_state_info_seir=state_info_seirs , 
    initial_theta_info_seir=theta_info_seirs , 
    observed_data=data[:days],
    num_state_particles=500,
    num_theta_particles=500,
    observation_distribution=obs_dist_negative_binomial,
    forecast_days=fday,
    N=N_pop
)



##########################################################################################################
########## SETP4: Visualize the Results ######Test_smc2##############################################################
# You can plot the filtered estimate of the state and parametersx
############################################################################################################

plt.style.use('seaborn-v0_8-white')

# Separation point for forecast
separation_point = data['Week ending'].iloc[-fday]

# --- MODEL WEIGHTS PLOT ---
w_dthp = smc2_results['weight_dthp']
w_seir = smc2_results['weight_seir']

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(data['Week ending'], w_dthp, label='Weight DTHP', color='orange', linewidth=2, alpha=0.8)
ax.plot(data['Week ending'], w_seir, label='Weight SEIR', color='dodgerblue', linewidth=2, alpha=0.8)
ax.axvline(x=separation_point, color='black', linestyle='--', linewidth=2)
ax.grid(True, linestyle='--', alpha=0.7)
# ax.set_facecolor('whitesmoke')
ax.set_xlabel('Date', fontsize=14, fontweight='bold')
ax.set_ylabel('Model weights', fontsize=16, fontweight='bold')
ax.set_title('Influenza', fontsize=18, fontweight='bold', pad=10)
ax.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
ax.xaxis.set_major_locator(MonthLocator(interval=3))
ax.xaxis.set_minor_locator(MonthLocator())
ax.xaxis.set_major_formatter(DateFormatter('%b %y'))
ax.tick_params(axis='both', which='major', labelsize=14)
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

#################################################################################
# --- Setup figure with 3 rows and custom layout ---
fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])  # Adjust height as needed

# Create axes
ax_inc = fig.add_subplot(gs[0, 0])     # SEIRS & DTHP Incidence
ax_rt = fig.add_subplot(gs[0, 1])      # SEIRS & DTHP Rt
ax_inc_ma = fig.add_subplot(gs[1, :])  # MA Incidence (full width)
ax_rt_ma = fig.add_subplot(gs[2, :])   # MA Rt (full width)

axes = [ax_inc, ax_rt, ax_inc_ma, ax_rt_ma]

# Add forecast and summer shading
for ax in axes:
    ax.axvspan(forecast_start, forecast_end, facecolor='lightgray', edgecolor='gray', alpha=0.25, linewidth=0.0, zorder=1)
    for start, end in summer_periods:
        ax.axvspan(start, end, color='thistle', alpha=0.2, zorder=0)

# --- Row 1: Incidence SEIRS + DTHP ---
for matrix_dict, label, color in matrices[:2]:
    plot_smc(matrix_dict['NI'], ax=ax_inc, Date=time, separation_point=separation_point, window=window, color=color, label=label)
ax_inc.scatter(time[before_sep], rolling_obs[before_sep], facecolor='black', marker='*', s=60, label='Observed Data (fit)', zorder=2)
ax_inc.scatter(time[after_sep], rolling_obs[after_sep], facecolor='brown', marker='*', s=60, label='Observed Data (forecast)', zorder=2)
ax_inc.axvline(x=separation_point, color='black', linestyle='--', linewidth=2)
ax_inc.set_ylabel("Incidence", fontsize=18, fontweight='bold')
ax_inc.legend(fontsize=12)
ax_inc.tick_params(axis='both', which='major', labelsize=12)
# ax_inc.set_ylim([-100, 6700])

# --- Row 1: Rt SEIRS + DTHP ---
for matrix_dict, label, color in matrices[:2]:
    plot_smc(matrix_dict['Rt'], ax=ax_rt, Date=time, separation_point=separation_point, window=window, color=color, label=label)
ax_rt.axhline(y=1, color='k', linestyle='--', linewidth=2)
ax_rt.set_ylabel(r"Reproduction number $R_t$", fontsize=18, fontweight='bold')
ax_rt.legend(fontsize=12)
ax_rt.tick_params(axis='both', which='major', labelsize=12)
ax_rt.set_ylim([0, 8])
# ax_rt.set_xlabel("Date", fontsize=14, fontweight='bold')

# --- Row 2: MA Incidence ---
plot_smc(ma_dict['NI'], ax=ax_inc_ma, Date=time, separation_point=separation_point, window=window, color=ma_color, label=ma_label, show_50ci=True)
ax_inc_ma.scatter(time[before_sep], rolling_obs[before_sep], facecolor='black', marker='*', s=60, label='Observed Data (fit)', zorder=2)
ax_inc_ma.scatter(time[after_sep], rolling_obs[after_sep], facecolor='brown', marker='*', s=60, label='Observed Data (forecast)', zorder=2)
ax_inc_ma.axvline(x=separation_point, color='black', linestyle='--', linewidth=2)
ax_inc_ma.set_ylabel("Incidence ", fontsize=18, fontweight='bold')
ax_inc_ma.legend(fontsize=12)
ax_inc_ma.tick_params(axis='both', which='major', labelsize=12)
# ax_inc_ma.set_ylim([-100, 6000])

# --- Row 3: MA Rt ---
plot_smc(ma_dict['Rt'], ax=ax_rt_ma, Date=time, separation_point=separation_point, window=window, color=ma_color, label=ma_label, show_50ci=True)
ax_rt_ma.axhline(y=1, color='k', linestyle='--', linewidth=2)
ax_rt_ma.set_ylabel(r"Reproduction number $R_t$", fontsize=18, fontweight='bold')
ax_rt_ma.set_xlabel("Date", fontsize=18, fontweight='bold')
ax_rt_ma.set_ylim([0, 8])
ax_rt_ma.legend(fontsize=12)
ax_rt_ma.tick_params(axis='both', which='major', labelsize=12)

# Final formatting
for ax in axes:
    ax.grid(True, linestyle='--', alpha=0.9)

plt.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.show()

###################################################################################################
### Parameters estimate #######################################################################

# --- PARAMETER TRAJECTORIES ---
L1 = [r'$\omega$', r'$\nu_{1}$', r'$\phi_1$']
L2 = [r'$\sigma$', r'$\gamma$', r'$\nu_{2}$', r'$\phi_2$']
nrows, ncols = 3, 3
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 12))
axes = axes.flatten()

# Plot DTHP parameters
for i, (state, matrix) in enumerate(matrix_dict_theta_dthp.items()):
    ax = axes[i]
    plot_smc(matrix, ax=ax, show_50ci=True)
    ci_025 = np.percentile(matrix[:, -1], 2.5)
    ci_975 = np.percentile(matrix[:, -1], 97.5)
    median_estimate = np.mean(matrix[:, -1])
    ax.set_title(f'{L1[i]}= {median_estimate:.3f} (95%CrI: [{ci_025:.3f}, {ci_975:.3f}])',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel(L1[i], fontsize=18)
    ax.set_xlabel('Time (days)', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.9)
    # ax.set_facecolor('whitesmoke')

# Plot SEIR parameters
offset = len(matrix_dict_theta_dthp)
for i, (state, matrix) in enumerate(matrix_dict_theta_seir.items()):
    ax = axes[offset + i]
    plot_smc(matrix, ax=ax, show_50ci=True)
    ci_025 = np.percentile(matrix[:, -1], 2.5)
    ci_975 = np.percentile(matrix[:, -1], 97.5)
    median_estimate = np.mean(matrix[:, -1])
    ax.set_title(f'{L2[i]}= {median_estimate:.3f} (95%CrI: [{ci_025:.3f}, {ci_975:.3f}])',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel(L2[i], fontsize=18)
    ax.set_xlabel('Time (days)', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.9)
    # ax.set_facecolor('whitesmoke')

# Hide unused subplots
for i in range(offset + len(matrix_dict_theta_seir), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

