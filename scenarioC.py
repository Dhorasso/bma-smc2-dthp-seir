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
import seaborn as sns
from scipy.stats import poisson, norm, nbinom
from numpy.random import binomial, normal
from joblib import Parallel, delayed  # For parallel computing
from plotnine import *
from tqdm import tqdm

# SMC2 Libraries
from models import stochastic_sir_model, dthp_model
from smc_squared import BMA_SMC2
from smc_visualization import trace_smc, plot_smc
# Style Configuration


############  SEPTP 1:Import/create your dataset ###########################
#### Generate the simulated data with time varying beta ###############
#######################################################################


def sir_var_beta(y, theta, t, dt=1):
    # Unpack variables
    S, I, R, NI, B = y
    N = S + I + R

    # Unpack parameters
    gamma = theta[0]

    # Transition probabilities
    P_SI = 1 - np.exp(-B * I/N * dt)       # Probability of transition from S to I
    P_IR = 1 - np.exp(-gamma * dt)         # Probability of transition from I to R

    # Binomial distributions for transitions
    B_SI = binomial(S, P_SI)
    B_IR = binomial(I, P_IR)

    # Update the compartments
    S -= B_SI
    I += B_SI - B_IR
    R += B_IR
    B = (np.exp(np.cos(2*np.pi*t/60)-t/100)) * 0.27
    NI = B_SI

    y_next = [max(0, compartment) for compartment in [S, I, R, NI, B]] # Ensure non-negative elements
    return y_next

def solve_sir_var_beta(model, theta, InitialState, t_start, t_end, dt=1):
    t_values = np.arange(t_start, t_end + dt, dt)
    results = np.zeros((len(t_values), len(InitialState)))

    # Set initial conditions
    results[0] = InitialState

    # Solve using Euler method
    for i in range(1, len(t_values)):
        results[i] = model(results[i - 1], theta, i, dt)

    # Convert to DataFrame for easy handling
    results_df = pd.DataFrame(results, columns=['S', 'I', 'R', 'NI', 'B'])
    results_df['time'] = t_values
    results_df['obs'] = np.random.poisson(results_df['NI'])
    return results_df

# Initial conditions
t = 0
B = (np.exp(np.cos(2*np.pi*t/59)-t/90)) * 0.27
true_theta = [1/6]  # true parameters
InitialState = [500000-1, 1, 0, 0, B]
t_start = 0
t_end = 130
dt = 1

np.random.seed(123) # Set a seed for reproducibility
simulated_data = solve_sir_var_beta(sir_var_beta, true_theta, InitialState, t_start, t_end, dt)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# First subplot: Plot new infections
axes[0].plot(simulated_data['time'].index, simulated_data['obs'], label='New Infected')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Population')
axes[0].legend()
axes[0].grid(True)
axes[0].set_title('New Infections Over Time')

# Second subplot: Plot transmission rate
axes[1].plot(simulated_data['time'].index, simulated_data['B'], label='Transmission Rate', color='orange')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Transmission Rate ')
axes[1].legend()
axes[1].grid(True)

# Adjust layout to avoid overlapping elements
plt.tight_layout()

# Show the plots
plt.show()
simulated_data


#################################################################################################
############ SEPTP 2: Define your observation distribution ######################################
# The observation_dist.py contains some examples, you extend to incorporate multiple dataste
#################################################################################################
# Poisson log-likelihood
def obs_dist_poisson(observed_data, model_data, theta, theta_names):
    epsi = 1e-4
    model_est_case = np.maximum(epsi, model_data['NI'].to_numpy())
    log_likelihoods = poisson.logpmf(observed_data['obs'], mu=model_est_case)
    log_likelihoods[np.isnan(log_likelihoods) | np.isinf(log_likelihoods)] = -np.inf
    return log_likelihoods

#################################################################################################
############ SEPTP 3: Run the SMC^2 #####################################################################
# You need to defined initial conditions for the state and prior for the parameter you want to estimate
##########################################################################################################

np.random.seed(123) # Set a seed for reproducibility

# # ##### # setting state and parameter

### SIR initial state and prior distribution
N_pop=2e5
state_info_seir = {
    'S': {'prior': [N_pop-3, N_pop, 0, 0, 'uniform']},  # Susceptibles
    'I': {'prior': [0, 3, 0, 0, 'uniform']},  # Infected
    'R': {'prior': [0, 0, 0, 0, 'uniform']},  # Removed (Recovered or Deceased)
    'NI': {'prior': [0, 0, 0,0, 'uniform']},  # Newly Infected
    'B': {'prior': [0, np.inf, 0.725, 0.01, 'normal']},  # Transmission rate (Beta)
}

theta_info_seir = {
    'gamma': {'prior': [0.1, 0.2, 0.16, 0.1,'truncnorm','log']},  # Removal rate (Inverse of infectious period)
    'nu_beta': {'prior': [0.05, 0.15,0.1,0.05, 'truncnorm','log']},  # Standard deviation of RW process (Beta variability)
    'phi': {'prior': [1e-5, 0.1,0,0, 'uniform','log']}  # Overdispersion parameter
}

# DTHP model state and parameter information
state_info_dthp = {
    'NI': {'prior': [0, 3, 0,0, 'uniform']},  # Newly Infected (lamabda_H(0))
    'C_I': {'prior': [0, 0, 0,0, 'uniform']},  # Cumulative Infected
    'Rt': {'prior': [0, np.inf, 4.25, 0.15, 'normal']}  # Reproduction Number
}

theta_info_dthp = {
    'omega_I': {'prior': [0.1, 0.2,0.16, 0.1,'truncnorm','log']},  # Decay parameter in the tiggering kernel
    'nu_beta': {'prior': [0.05,0.15,0.1,0.05, 'uniform', 'log']},  # Standard deviation of RW process (Beta variability)
    'phi': {'prior': [0.001, 0.1,0,0, 'uniform','log']}  # Overdispersion parameter
}



days=t_end-14
fday=14

smc2_results = BMA_SMC2(
    model_sir=stochastic_sir_model,
    model_dthp= dthp_model_flu, 
    initial_state_info_dthp=state_info_dthp , 
    initial_theta_info_dthp=theta_info_dthp , 
    model_seir=stochastic_model_sir, 
    initial_state_info_seir=state_info_seir , 
    initial_theta_info_seir=theta_info_seir , 
    observed_data=simulated_data[:days+1],
    num_state_particles=200,
    num_theta_particles=400,
    observation_distribution=obs_dist_poisson,
    forecast_days=fday,
)


# Print the Marginal log-likelihood
print("Marginal log-likelihood:", smc2_results['margLogLike'])

##########################################################################################################
########## SETP5: Visualize the Results ####################################################################
# You can plot the filtered estimate of the state and parametersx
############################################################################################################

# state trajectory particles and extract corresponding matrix

trajParticles_state = smc2_results['trajState']
matrix_dict_state = trace_smc(trajParticles_state)


trajParticles_theta = smc2_results['trajtheta']
matrix_dict_theta = trace_smc(trajParticles_theta)
g=np.median(matrix_dict_theta['gamma'][:,-1])

# Calculate the  reproduction number Rt and add it to the satate dict
matrix_dict_state['Rt']=matrix_dict_state['B']/g


# simulated_data['Rt'] =simulated_data['Bt']/(1/7)

import matplotlib.pyplot as plt

# Create a figure with a 2-row, 1-column subplot layout
fig, axs = plt.subplots(1, 2, figsize=(16, 5))  # Adjusted figsize for two rows

# First subplot for 'NI', the number of new infected
for (state, matrix) in matrix_dict_state.items():
    if state == 'NI':
        # Plot the SMC results in the first subplot
        plot_smc(matrix, ax=axs[0])
        # plot_smc2(matrix_dict_full[state], ax=axs[0])

        # Plot observed data points (similar to geom_point)
        axs[0].scatter(simulated_data['time'], simulated_data['obs'], color='darkorange', edgecolor='salmon', s=30, label='Observed Data')

        # Customize labels and axis for 'NI'
        axs[0].set_xlabel('Time (days)', fontsize=18, fontweight='bold')
        axs[0].set_ylabel('Daily new cases', fontsize=18, fontweight='bold')
        
        # Show legend and grid for 'NI'
        axs[0].legend(loc='upper left', fontsize=14)

# Second subplot for 'B', the transmission rate
for (state, matrix) in matrix_dict_state.items():
    if state == 'B':
        # Plot the SMC results in the second subplot
        plot_smc(matrix, ax=axs[1])
        # plot_smc2(matrix_dict_full[state], ax=axs[1])

        # Add a horizontal dashed line at y=1 (or the specific line related to 'B')
        axs[1].plot(simulated_data['time'], simulated_data['B'], color='orange', linestyle='--', lw=4, label='Truth')

        # Customize labels and axis for 'B'
        axs[1].set_xlabel('Time (days)', fontsize=18, fontweight='bold')
        axs[1].set_ylabel(r' $\beta_t$', fontsize=18, fontweight='bold')

        # Show legend and grid for 'B'
        axs[1].legend(loc='upper right', fontsize=14)

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the entire figure with both subplots
plt.show()


#############################################################################
# Plot for the parameter trajectories
##############################################################################

# Example of labels 
L = [ r'$\sigma$', r'$\gamma$', r'$\nu_{\beta}$']

data_for_corner = []
labels = []

# Collect data and labels for each parameter
for i, (state, matrix) in enumerate(matrix_dict_theta.items()):
    data = matrix[:, -1]  # Using the last column of each matrix
    data_for_corner.append(data)
    labels.append(L[i])

data_for_corner = np.column_stack(data_for_corner)

# Create a 3x3 subplot layout
N=len(matrix_dict_theta) 
fig, axs = plt.subplots(N-1, N, figsize=(18, 8))

# Row 1: Plot time series using `plot_smc(matrix)` for each parameter
for i, (state, matrix) in enumerate(matrix_dict_theta.items()):
    # plot_smc2(matrix_dict_full[state], ax=axs[0, i])
    plot_smc(matrix, ax=axs[0, i])  # Assuming plot_smc returns data suitable for plotting
    # axs[0, i].set_title(f'Time Series of {L[i]}', fontsize=14)
    axs[0, i].set_xlabel('Time (days)', fontsize=18, fontweight='bold')
    axs[0, i].set_ylabel(L[i],  fontsize=18, fontweight='bold')
    
    # Add a horizontal line for the true parameter values
    if i<2:
        axs[0, i].axhline(y=true_theta[i], color='orange', linestyle='--', linewidth=3, label='True Value')
axs[0, 1].legend(loc='upper left')

# Row 2: Histograms for each parameter
for idx, label in enumerate(labels):
    axs[1, idx].hist(data_for_corner[:, idx], bins=20, density=True, color='navy', alpha=0.4, lw=5)
    sns.kdeplot(data_for_corner[:, idx], ax=axs[1, idx], color='dodgerblue', lw=3)
    
    # Calculate 0.25 and 0.975 CIs
    ci_025 = np.percentile(data_for_corner[:, idx], 2.5)
    ci_975 = np.percentile(data_for_corner[:, idx], 97.5)
    median_estimate = np.median(data_for_corner[:, idx])
    
    # Set title including the median value
    axs[1, idx].set_title(f'{label}= {median_estimate:.3f} (95%CrI: [{ci_025:.3f}, {ci_975:.3f}])', fontsize=14, fontweight='bold')
    
    axs[1, idx].set_xlabel(L[idx], fontsize=18, fontweight='bold')
    axs[1, idx].set_ylabel('Density', fontsize=18, fontweight='bold')
    
    # Add vertical line for true value and median
    if idx<2:
        axs[1, idx].axvline(true_theta[idx], color='orange', linestyle='--', linewidth=3, label='True Value')
    axs[1, idx].axvline(median_estimate, color='k', linewidth=3, label='Median')
    
    # Add dashed lines for 0.25 and 0.975 CIs
    axs[1, idx].axvline(ci_025, color='dodgerblue', linestyle='--', linewidth=2)
    axs[1, idx].axvline(ci_975, color='dodgerblue', linestyle='--', linewidth=2)
    
    # Add legend
axs[1, 1].legend()

plt.tight_layout()
plt.show()
