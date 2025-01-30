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
from numpy.random import binomial, normal
from joblib import Parallel, delayed  # For parallel computing
from tqdm import tqdm


# SMC2 Libraries
from models import stochastic_sir_model, dthp_model
from smc_squared import BMA_SMC2
from smc_visualization import trace_smc, plot_smc, compute_model_average
# Style Configuration
plt.style.use('seaborn-v0_8-white')

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
true_theta = [1/6]  # true value of gamma
InitialState = [500000-1, 1, 0, 0, B]
t_start = 0
t_end = 130
dt = 1

np.random.seed(123) # Set a seed for reproducibility
simulated_data = solve_sir_var_beta(sir_var_beta, true_theta, InitialState, t_start, t_end, dt)
simulated_data['Rt'] =simulated_data['B']/(1/6) # reproduction number
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# First subplot: Plot new infections
axes[0].plot(simulated_data['time'].index, simulated_data['obs'], label='New Infected')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Population')
axes[0].legend()
axes[0].grid(True)
axes[0].set_title('New Infections Over Time')

# Second subplot: Plot transmission rate
axes[1].plot(simulated_data['time'].index, simulated_data['Rt'], label='Reproduction nmber', color='orange')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Reproduction nmber ')
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

# Negative Binomial log-likelihood
def obs_dist_negative_binomial(observed_data, model_data, theta, theta_names):
    epsi = 1e-4
    model_est_case = np.maximum(epsi, model_data['NI'].to_numpy())
    param = dict(zip(theta_names, theta))
    overdispersion = param.get('phi', 0.1)  # Default value for 'phi' if not provided
    log_likelihoods = nbinom.logpmf(observed_data['obs'], 1 / overdispersion, 1 / (1 + overdispersion * model_est_case))
    log_likelihoods[np.isnan(log_likelihoods) | np.isinf(log_likelihoods)] = -np.inf
    return log_likelihoods

#################################################################################################
############ SEPTP 3: Run the SMC^2 #####################################################################
# You need to defined initial conditions for the state and prior for the parameter you want to estimate
##########################################################################################################

np.random.seed(123) # Set a seed for reproducibility

# # ##### # setting state and parameter

### SIR initial state and prior distribution
N_pop=5e5
state_info_sir = {
    'S': {'prior': [N_pop-3, N_pop, 0, 0, 'uniform']},  # Susceptibles
    'I': {'prior': [0, 3, 0, 0, 'uniform']},  # Infected
    'R': {'prior': [0, 0, 0, 0, 'uniform']},  # Removed (Recovered or Deceased)
    'NI': {'prior': [0, 0, 0,0, 'uniform']},  # Newly Infected
    'B': {'prior': [0, np.inf, 0.725, 0.01, 'normal']},  # Transmission rate (Beta)
}

theta_info_sir = {
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



fday=14
days=len(simulated_data)-fday


smc2_results = BMA_SMC2(
    model_sir=stochastic_sir_model,
    model_dthp= dthp_model, 
    initial_state_info_dthp=state_info_dthp , 
    initial_theta_info_dthp=theta_info_dthp , 
    initial_state_info_sir=state_info_sir , 
    initial_theta_info_sir=theta_info_sir , 
    observed_data=simulated_data[:days],
    num_state_particles=200,
    num_theta_particles=400,
    observation_distribution=obs_dist_negative_binomial,
    forecast_days=fday,
)




##########################################################################################################
########## SETP4: Visualize the Results ######Test_smc2##############################################################
# You can plot the filtered estimate of the state and parametersx
############################################################################################################

separation_point = simulated_data['time'].iloc[-1]-fday
###  Evolution of the model weights
w_dthp = smc2_results['weight_dthp']  
w_sir = smc2_results['weight_sir']

# Plotting
fig, ax = plt.subplots(figsize=(17/3, 4))  
ax.plot(w_dthp, label='Weight dthp', color='orange', linewidth=2, alpha=0.8)
ax.plot(w_sir, label='Weight sir', color='dodgerblue', linewidth=2, alpha=0.8)
ax.grid(True, linestyle='--', alpha=0.7)  # Add grid with dashed lines
ax.set_facecolor('whitesmoke')  # Add light background color to the plot area
# Adding labels and title with improved fonts
ax.set_xlabel('Time (days)', fontsize=18, fontweight='bold', labelpad=10)
ax.set_ylabel('Model weights', fontsize=18, fontweight='bold', labelpad=10)
ax.set_title('Scenario C', fontsize=18, fontweight='bold', pad=15)
# Customize ticks
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=10)
ax.axvline(x=separation_point, color='black', linestyle='--', linewidth=2)
ax.legend(fontsize=12, loc='upper right', frameon=True, shadow=True, facecolor='white')
# Tight layout for better spacing
fig.tight_layout()
# Display the plot
plt.show()


###################################################################
# state trajectory particles and extract corresponding matrix



# Extract trajectories SIR model
trajParticles_state_sir = smc2_results['traj_state_sir']
matrix_dict_state_sir = trace_smc(trajParticles_state_sir)
trajParticles_theta_sir = smc2_results['traj_theta_sir']
matrix_dict_theta_sir = trace_smc(trajParticles_theta_sir)
gamma=np.mean(matrix_dict_theta_sir['gamma'][:,-1])
# Calculate the  reproduction number Rt and add it to the satate dict
matrix_dict_state_sir['Rt']=matrix_dict_state_sir['B']/gamma

# Extract trajectories DTHP
trajParticles_state_dthp = smc2_results['traj_state_dthp']
matrix_dict_state_dthp = trace_smc(trajParticles_state_dthp)
trajParticles_theta_dthp = smc2_results['traj_theta_dthp']
matrix_dict_theta_dthp = trace_smc(trajParticles_theta_dthp)

# Compute trajectories model-averging based on the model weights
matrix_dict_state_avg = compute_model_average(matrix_dict_state_dthp, matrix_dict_state_sir, w_dthp, w_sir)


# Plot incidence and time-varying reproduction number for the 3 models
fig, axs = plt.subplots(2, 3, figsize=(17, 10), sharex=True, sharey='row')
# Titles for the subplots
titles = ['DTHP', 'SIR', 'MA']

# Plot NI (top row)
window = 1 ## increase if you want more smoothed plot
for i, (method, matrix) in enumerate([
    ('DTHP', matrix_dict_state_dthp['NI']),
    ('SIR', matrix_dict_state_sir['NI']),
    ('MA', matrix_dict_state_avg['NI'])
]):
    plot_smc(matrix, ax=axs[0, i], separation_point=separation_point,  window=window)
    # Add observed data for days 1 to `days`
    axs[0, i].scatter(
        simulated_data['time'][:days], 
        simulated_data['obs'][:days].rolling(window=window, min_periods=1).mean(), 
        facecolor='yellow', edgecolor='salmon', s=20, label='Observed Data (Fitted)', zorder=2
    )
    
    # Add observed data for days > `days`
    axs[0, i].scatter(
        simulated_data['time'][days:], 
        simulated_data['obs'][days:].rolling(window=window, min_periods=1).mean(), 
        facecolor='orange', edgecolor='salmon', s=20, label='Observed Data (Forecast)', zorder=2
    )
    
    # Add subplot title
    axs[0, i].set_title(f'{titles[i]}', fontsize=20, fontweight='bold')
    axs[0, i].axvline(x=separation_point, color='black', linestyle='--', linewidth=2, label=f'Separation (t={separation_point})')
    
    # Add Y-axis label for the first column
    if i == 0:
        axs[0, i].set_ylabel('Incidence', fontsize=18, fontweight='bold')
    
    # Customize tick parameters
    axs[0, i].tick_params(axis='both', which='major', labelsize=16)
    # axs[0, i].set_ylim([-10, 180])

# Plot Rt (bottom row)
for i, (method, matrix) in enumerate([
    ('DTHP', matrix_dict_state_dthp['Rt']),
    ('SIR', matrix_dict_state_sir['Rt']),
    ('MA', matrix_dict_state_avg['Rt'])
]):
    # Plot SMC results
    plot_smc(matrix, ax=axs[1, i],separation_point=separation_point,  window=window)
    # Plot true Rt curve
    axs[1, i].plot(
        simulated_data['time'], 
        simulated_data['Rt'].rolling(window=window, min_periods=1).mean(), 
        color='gold', lw=4, linestyle='--', label='True $R_t$', zorder=3
    )
    
    # Add horizontal line for Rt = 1
    axs[1, i].axhline(y=1, color='k', linestyle='--', linewidth=2, label=r'$R_t = 1$', zorder=1)

    # Add Y-axis label for the first column
    if i == 0:
        axs[1, i].set_ylabel(r'Reproduction number $R_t$', fontsize=18, fontweight='bold')
    
    # Add X-axis label for the bottom row
    axs[1, i].set_xlabel('Time (days)', fontsize=16, fontweight='bold')
    axs[1, i].set_ylim([0, 6])
    axs[1, i].tick_params(axis='both', which='major', labelsize=16)
    axs[1, i].axvline(x=separation_point, color='black', linestyle='--', linewidth=2)

# Adjust layout to prevent overlapping and add horizontal spacing
# Create legend elements
legend_elements = [
    mpatches.Patch(facecolor='steelblue', label='Estimate'),        # Steelblue patch
    mpatches.Patch(facecolor='mediumpurple', label='Forecast'),     # Mediumpurple patch
    mlines.Line2D([], [], color='gold', lw=3, linestyle='--',label='True $R_t$'),  # Gold line for True Rt
    mlines.Line2D([], [], marker='o', color='salmon', markerfacecolor='yellow', 
                  markersize=10, linestyle='None', label='Observed Data (Fitted)'),  # Yellow dots with salmon edge
    mlines.Line2D([], [], marker='o', color='salmon', markerfacecolor='orange', 
                  markersize=10, linestyle='None', label='Observed Data (Forecast)')  # Orange dots with salmon edge
]
# Add legend below the plots
fig.legend(
    handles=legend_elements, loc='lower center', ncol=5, fontsize=20, frameon=False, bbox_to_anchor=(0.5, -0.02)
)
# Improve overall appearance: grid, tight layout, etc.
for ax in axs.flat:
    ax.grid(True, linestyle='--', alpha=0.9)  # Add grid with dashed lines
    ax.set_facecolor('whitesmoke')  # Add background color for each subplot
# Show the plot
plt.tight_layout(rect=[0, 0.04, 1, 0.95])  # Adjust layout to leave space for the legend
# plt.show()
plt.subplots_adjust(wspace=0.05, hspace=0.1)




#############################################################################
# Plot for the parameter trajectories
##############################################################################


# Example label
L1 = [r'$\omega$', r'$\nu_{1}$', r'$\phi_1$']
L2 = [r'$\gamma$', r'$\nu_{2}$', r'$\phi_2$']

# Data sources

# Combined number of rows and columns for subplots
nrows, ncols = 2, 3
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 8))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Iterate and plot for the first dataset
for i, (state, matrix) in enumerate(matrix_dict_theta_dthp.items()):
    ax = axes[i]  # Current axis
    plot_smc(matrix, ax=ax)   
    # Calculate 0.25 and 0.975 CIs
    ci_025 = np.percentile(matrix[:, -1], 2.5)
    ci_975 = np.percentile(matrix[:, -1], 97.5)
    median_estimate = np.mean(matrix[:, -1])
    ax.set_title(f'{L1[i]}= {median_estimate:.3f} (95%CrI: [{ci_025:.3f}, {ci_975:.3f}])', fontsize=18, fontweight='bold')
    ax.set_ylabel(L1[i], fontsize=25, fontweight='bold')
# Iterate and plot for the second dataset
for i, (state, matrix) in enumerate(matrix_dict_theta_sir.items()):
    ax = axes[i + 3]  # Move to the second row
    plot_smc(matrix, ax=ax)
    # Calculate 0.25 and 0.975 CIs
    ci_025 = np.percentile(matrix[:, -1], 2.5)
    ci_975 = np.percentile(matrix[:, -1], 97.5)
    median_estimate = np.mean(matrix[:, -1])
    # Set title including the median value
    ax.set_title(f'{L2[i]}= {median_estimate:.3f} (95%CrI: [{ci_025:.3f}, {ci_975:.3f}])', fontsize=18, fontweight='bold')
    # Add a true value line for the first state in the second dataset if applicable
    if i == 0:
        ax.axhline(true_theta[i], color='orange', linestyle='--', linewidth=3, label='True Value')
        # ax.legend(fontsize=12)
    # Customize labels
    ax.set_xlabel('Time (days)', fontsize=16, fontweight='bold')
    ax.set_ylabel(L2[i], fontsize=25, fontweight='bold')
# Improve overall appearance: grid, tight layout, etc.
fig.text(0.02, 0.97, 'C', fontsize=50, fontweight='bold', ha='center', va='center')
# Improve overall appearance: grid, tight layout, etc.
for ax in axes.flat:
    ax.grid(True, linestyle='--', alpha=0.9)  # Add grid with dashed lines
    ax.set_facecolor('whitesmoke')  # Add background color for each subplot
# Adjust layout and show the complete figure
plt.tight_layout()
plt.show()

