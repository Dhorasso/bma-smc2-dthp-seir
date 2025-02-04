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
from models import stochastic_sirs_model, dthp_model
from smc_squared import BMA_SMC2
from smc_visualization import trace_smc, plot_smc, compute_model_average
# Style Configuration
plt.style.use('seaborn-v0_8-white')

############  SEPTP 1:Import your dataset ###########################
# Assuming the uploaded file is named "COVID-19_HPSC_Detailed_Statistics_Profile.csv"
#######################################################################


file_path = r"Influenza2024.xlsx"

# Read the CSV file into a pandas DataFrame
df = pd.read_excel(file_path)
# Restrict the observations from 29 May, 2022
data = df.iloc[72:].copy()
data =  data.reset_index(drop=True)
# Ensure the 'Date' column is in datetime format
data['Week ending'] = pd.to_datetime(data['Week ending'])
data['obs']= data['Number of influenza cases'] 
# Plotting with matplotlib
fig, ax = plt.subplots(figsize=(9, 4))

# Plot the Confirmed Covid Cases
ax.plot(data['Week ending'], data['Number of influenza cases'], color='blue')

# Formatting the plot
# ax.set_title("COVID-19 Confirmed Cases Over Time", fontsize=16)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Weekly Cases", fontsize=12)
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

### SIRS initial state and prior distribution
N_pop = 5.16e6  # Total population
I0 = 15  # Initial number of infected individuals

state_info_sirs = {
    'S': {'prior': [N_pop - I0, N_pop, 0, 0, 'uniform']},  # Susceptibles
    'I': {'prior': [0, I0, 0, 0, 'uniform']},  # Infected
    'R': {'prior': [0, 0, 0, 0, 'uniform']},  # Recovered
    'NI': {'prior': [0, 0, 0, 0, 'uniform']},  # Newly Infected
    'B': {'prior': [0, 1, 0.45, 0.05, 'normal']},  # Transmission rate (Beta)
}

theta_info_sirs = {
    'gamma': {'prior': [0, 1, 0.7, 0.2, 'truncnorm', 'log']},  # Recovery rate
    'alpha': {'prior': [1 / (12 * 4), 1 / (6 * 4), 1 / 2, 0.1, 'uniform', 'log']},  # Transition from R to S
    'nu_beta': {'prior': [0.05, 0.2, 0.1, 0.05, 'uniform', 'log']},  # Volatility in transmission rate
    'phi': {'prior': [0.001, 0.3, 0, 0, 'uniform', 'log']},  # Overdispersion parameter
}

### DTHP model state and prior distribution
state_info_dthp = {
    'NI': {'prior': [0, I0, 0, 0, 'uniform']},  # Newly Infected (lambda_H(0))
    'C_I': {'prior': [0, 0, 0, 0, 'uniform']},  # Cumulative Infected
    'Rt': {'prior': [0, np.inf, 0.5, 0.05, 'normal']},  # Reproduction Number
}

theta_info_dthp = {
    'omega_I': {'prior': [0, 1, 0.5, 0.2, 'truncnorm', 'log']},  # Decay parameter in the triggering kernel
    'nu_beta': {'prior': [0.05, 0.2, 0.1, 0.05, 'uniform', 'log']},  # Volatility in transmission rate
    'phi': {'prior': [0.001, 0.3, 0, 0, 'uniform', 'log']},  # Overdispersion parameter
}



fday=16
days=len(data)-fday

np.random.seed(123) # Set a seed for reproducibility
smc2_results = BMA_SMC2(
    model_sir=stochastic_sirs_model,
    model_dthp= dthp_model, 
    initial_state_info_dthp=state_info_dthp , 
    initial_theta_info_dthp=theta_info_dthp , 
    initial_state_info_sir=state_info_sirs , 
    initial_theta_info_sir=theta_info_sirs , 
    observed_data=data[:days],
    num_state_particles=500,
    num_theta_particles=1000,
    observation_distribution=obs_dist_negative_binomial,
    forecast_days=fday,
    N=N_pop
)




##########################################################################################################
########## SETP4: Visualize the Results ######Test_smc2##############################################################
# You can plot the filtered estimate of the state and parametersx
############################################################################################################

separation_point = data['Week ending'].iloc[-fday]
###  Evolution of the model weights
w_dthp = smc2_results['weight_dthp']  
w_sir = smc2_results['weight_sir']

# Plotting
fig, ax = plt.subplots(figsize=(9, 4))  
ax.plot(data['Week ending'], w_dthp, label='Weight dthp', color='orange', linewidth=2, alpha=0.8)
ax.plot(data['Week ending'], w_sir, label='Weight sir', color='dodgerblue', linewidth=2, alpha=0.8)
ax.grid(True, linestyle='--', alpha=0.7)  # Add grid with dashed lines
ax.set_facecolor('whitesmoke')  # Add light background color to the plot area
# Adding labels and title with improved fonts
ax.set_xlabel('Date', fontsize=18, fontweight='bold', labelpad=10)
ax.set_ylabel('Model weights', fontsize=18, fontweight='bold', labelpad=10)
ax.axvline(x=separation_point, color='black', linestyle='--', linewidth=2)
ax.legend(fontsize=12, loc='upper right', frameon=True, shadow=True, facecolor='white')
ax.xaxis.set_major_locator(MonthLocator(interval=3))  # Major ticks every 3 months
ax.xaxis.set_minor_locator(MonthLocator())  # Minor ticks every month
ax.xaxis.set_major_formatter(DateFormatter('%b %y'))  
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

#################################################################################
# Model averaging plots



# Create a figure with a 2-row, 1-column subplot layout
fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # Adjusted figsize for two rows

# First subplot for 'NI'
for (state, matrix) in matrix_dict_state_avg.items():
    if state == 'NI':
        plot_smc(matrix, ax=axs[0], Date=data['Week ending'],separation_point=separation_point)
        axs[0].scatter(data['Week ending'][:days], data['obs'][:days], color='yellow', edgecolor='salmon',\
                       s=20, label='Observed Data (Fitting)', zorder=2)
        axs[0].scatter(data['Week ending'][days:], data['obs'][days:], color='orange', edgecolor='salmon',\
                       s=20, label='Observed Data (Forecast)', zorder=2)
        axs[0].set_ylabel('Incidence', fontsize=18, fontweight='bold')

# Second subplot for the reproduction number
window=1
for (state, matrix) in matrix_dict_state_avg.items():
    if state == 'Rt':
        plot_smc(matrix, ax=axs[1], Date=data['Week ending'], separation_point=separation_point, window=window)
        axs[1].axhline(y=1, color='k', linestyle='--', linewidth=2, label=r'$R_t = 1$', zorder=1)
        axs[1].set_ylabel(r'Reproduction number $R_t$', fontsize=16, fontweight='bold')
        axs[1].set_ylim([0.01,2.5])

legend_elements = [
    mpatches.Patch(facecolor='steelblue', label='Estimate'),        # Steelblue patch
    mpatches.Patch(facecolor='mediumpurple', label='Forecast'),     # Mediumpurple patch
    # mlines.Line2D([], [], color='gold', lw=3, linestyle='--',label='True $R_t$'),  # Gold line for True Rt
    mlines.Line2D([], [], marker='o', color='salmon', markerfacecolor='yellow', 
                  markersize=10, linestyle='None', label='Observed Data (Fitted)'),  # Yellow dots with salmon edge
    mlines.Line2D([], [], marker='o', color='salmon', markerfacecolor='orange', 
                  markersize=10, linestyle='None', label='Observed Data (Forecast)')  # Orange dots with salmon edge
]

# Add legend below the plots

fig.legend(
    handles=legend_elements, loc='lower center', ncol=5, fontsize=14, frameon=False, bbox_to_anchor=(0.5, -0.02)
)

plt.tight_layout(rect=[0, 0.04, 1, 0.95])  # Adjust layout to leave space for the legend
# plt.show()
plt.subplots_adjust(wspace=0.05, hspace=0.3)
plt.show()



######################################################################################
#### DTHP and SIRS model plots ####################################################
fig, axs = plt.subplots(2, 2, figsize=(16, 8))  # Adjusted figsize for better layout

# Plotting for the first row (Incidence)
# DTHP Incidence
for (state, matrix) in matrix_dict_state_dthp.items():
    if state == 'NI':
        plot_smc(matrix, Date=data['Week ending'], ax=axs[0, 0], separation_point=separation_point)
        axs[0, 0].scatter(data['Week ending'][:days], data['obs'][:days], color='yellow', edgecolor='salmon', \
                          s=20, label='Observed Data (Fitting)', zorder=2)
        axs[0, 0].scatter(data['Week ending'][days:], data['obs'][days:], color='orange', edgecolor='salmon',\
                          s=20, label='Observed Data (Forecast)', zorder=2)
        axs[0, 0].set_ylabel('Incidence', fontsize=18, fontweight='bold')
        axs[0, 0].set_title('DTHP', fontsize=20, fontweight='bold')

# SIRS Incidence
for (state, matrix) in matrix_dict_state_sir.items():
    if state == 'NI':
        plot_smc(matrix, Date=data['Week ending'], ax=axs[0, 1], separation_point=separation_point)
        axs[0, 1].scatter(data['Week ending'][:days], data['obs'][:days], color='yellow', edgecolor='salmon', \
                          s=20, label='Observed Data (Fitting)', zorder=2)
        axs[0, 1].scatter(data['Week ending'][days:], data['obs'][days:], color='orange', edgecolor='salmon', \
                          s=20, label='Observed Data (Forecast)', zorder=2)
        axs[0, 1].set_title('SIRS', fontsize=20, fontweight='bold')

# Plotting for the second row (Reproduction Number)
# DTHP Reproduction Number
for (state, matrix) in matrix_dict_state_dthp.items():
    if state == 'Rt':
        plot_smc(matrix, Date=data['Week ending'], ax=axs[1, 0], separation_point=separation_point, window=window)
        axs[1, 0].axhline(y=1, color='k', linestyle='--', linewidth=2, label=r'$R_t = 1$', zorder=1)
        axs[1, 0].set_ylabel(r'Reproduction number $R_t$', fontsize=16, fontweight='bold')

# SIRS Reproduction Number
for (state, matrix) in matrix_dict_state_sir.items():
    if state == 'Rt':
        plot_smc(matrix, Date=data['Week ending'], ax=axs[1, 1], separation_point=separation_point, window=1)
        axs[1, 1].axhline(y=1, color='k', linestyle='--', linewidth=2, label=r'$R_t = 1$', zorder=1)

# Remove y-labels from the second column
for ax in [axs[0, 1], axs[1, 1]]:
    ax.set_ylabel('')

fig.legend(
    handles=legend_elements, loc='lower center', ncol=4, fontsize=16, frameon=False, bbox_to_anchor=(0.5, -0.02)
)

plt.tight_layout(rect=[0, 0.04, 1, 0.95])  # Adjust layout to leave space for the legend
# plt.show()
plt.subplots_adjust(wspace=0.1, hspace=0.3)
plt.show()


###################################################################################################
### Parameters estimate #######################################################################

# Placeholder for the state trajectories
L1 = [r'$\omega$', r'$\nu_{1}$', r'$\phi_1$']
L2 = [r'$\gamma$', r'$\alpha$', r'$\nu_{2}$', r'$\phi_2$']


# Combined number of rows and columns for subplots
total_plots = len(matrix_dict_theta_dthp) + len(matrix_dict_theta_sir)
nrows = (total_plots + 2) // 3  # Calculate rows dynamically based on total plots
ncols = 3
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 12))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Iterate and plot for the first dataset
for i, (state, matrix) in enumerate(matrix_dict_theta_dthp.items()):
    ax = axes[i]  # Current axis

    # Plot using the updated `plot_smc` function
    plot_smc(matrix, ax=ax)
        
    # Calculate 0.25 and 0.975 CIs
    ci_025 = np.percentile(matrix[:, -1], 2.5)
    ci_975 = np.percentile(matrix[:, -1], 97.5)
    median_estimate = np.mean(matrix[:, -1])
    
    # Set title including the median value
    ax.set_title(f'{L1[i]}= {median_estimate:.3f} (95%CrI: [{ci_025:.3f}, {ci_975:.3f}])', fontsize=18, fontweight='bold')
    
    # Customize labels
    ax.set_ylabel(L1[i], fontsize=25, fontweight='bold')
    ax.set_xlabel('Time (week)', fontsize=16, fontweight='bold')

# Iterate and plot for the second dataset
for i, (state, matrix) in enumerate(matrix_dict_theta_sir.items()):
    ax = axes[len(matrix_dict_theta_dthp) + i]  # Adjust index for second dataset

    # Plot using the updated `plot_smc` function
    # if state=='alpha' or state=='gamma':
    #     matrix=1/matrix
    plot_smc(matrix, ax=ax)
        
    # Calculate 0.25 and 0.975 CIs
    ci_025 = np.percentile(matrix[:, -1], 2.5)
    ci_975 = np.percentile(matrix[:, -1], 97.5)
    median_estimate = np.mean(matrix[:, -1])
    
    # Set title including the median value
    ax.set_title(f'{L2[i]}= {median_estimate:.3f} (95%CrI: [{ci_025:.3f}, {ci_975:.3f}])', fontsize=18, fontweight='bold')
    
    # Customize labels
    ax.set_xlabel('Time (week)', fontsize=16, fontweight='bold')
    ax.set_ylabel(L2[i], fontsize=25, fontweight='bold')

# Hide unused subplots if any
for i in range(total_plots, len(axes)):
    fig.delaxes(axes[i])

# Add text label for the figure

# Improve overall appearance: grid, tight layout, etc.
for ax in axes[:total_plots]:  # Only iterate over used axes
    ax.grid(True, linestyle='--', alpha=0.9)  # Add grid with dashed lines
    ax.set_facecolor('whitesmoke')  # Add background color for each subplot

# Adjust layout and show the complete figure
plt.tight_layout()
plt.show()
