###########################################################################
#  This file contains the code for different (SIR and DTHP)
#######################################################################

import pandas as pd
import numpy as np
from numpy.random import binomial, normal 


######################################################################################################
#####  SIR model with time-varying Beta as a geometric random walk #################################

def stochastic_sir_model(y, theta, theta_names, dt=1):
    """
    Vectorized discrete-time stochastic SIR compartmental model with time-varying Beta.
    
    Parameters:
    ----------
    y : np.ndarray
        A 2D array of compartments with shape (num_particles, num_compartments). 
        Columns represent the following compartments:
        [S (Susceptible), I (Infected), R (Removed), 
         NI (new infected), B (Transmission Rate)].
    theta : np.ndarray
        A 1D array of parameter values, one per particle. The parameters must match the order in `theta_names`.
    theta_names : list of str
        List of parameter names, matching the order in `theta`.
    dt : float, optional
        Time step for discrete updates (default is 1).

    Returns:
    -------
    np.ndarray
        Updated 2D array of compartments with the same shape as input `y`.
    """

    # Unpack compartments (columns of y)
    S, I, R, NI, B = y.T

    # Calculate total population for each particle
    N = S + I + R

    # Unpack parameters into a dictionary for easy access
    param = dict(zip(theta_names, theta))
    gamma = param['gamma']  # Recovery rate
    nu_beta = param.get('nu_beta', 0.1)  # Default value for `nu_beta` if not specified

    # Transition probabilities (vectorized)
    P_SI = 1 - np.exp(-B * I / N * dt)  # Susceptible → Infected
    P_IR = 1 - np.exp(-gamma * dt)      # Infected → Removed

    # Simulate transitions using binomial draws
    Y_SI = binomial(S.astype(int), P_SI)  # S → I
    Y_IR = binomial(I.astype(int), P_IR)  # I → R

    # Update compartments
    S_next = S - Y_SI
    I_next = I + Y_SI - Y_IR
    R_next = R + Y_IR
    
    # Update transmission rate with stochastic volatility
    B_next = B * np.exp(nu_beta * normal(0, 1, size=B.shape) * dt)

    # Update new infected
    NI_next = Y_SI

    # Combine updated compartments into a 2D array
    y_next = np.column_stack((S_next, I_next, R_next, NI_next, B_next))
    
    # Ensure all compartments remain non-negative
    return np.maximum(y_next, 0)


####################################################################
######### Discrete-time Hawkes Process    ########################

def dthp_model(state, theta, state_names, theta_names, observed_data, t):
    """
    Vectorized Discrete-time Hawkes Process for flu modeling.

    Parameters:
    - state: 2D array of state variables (num_particles x num_state_variables)
    - theta: 2D array of parameters (num_particles x num_parameters)
    - state_names: List of state variable names
    - theta_names: List of parameter names
    - observed_data: Dictionary containing observed data, including 'obs' key with array (num_time_steps)
    - t: Current time step (scalar)

    Returns:
    - updated_state: DataFrame of updated state variables with state names as columns
    """
    # Unpack state variables
    lm_I, C_I, Rt = state.T.copy()  # Ensure state variables are writable

    # Total population size
    N = 2e5

    # Create parameter dictionary
    param = dict(zip(theta_names, theta))  # Transpose theta for particle-wise parameter handling
    omega_I = param['omega_I']  # Infection kernel decay parameter
    nu_beta = param['nu_beta']  # Noise parameter for Rt

    # Reset lambdas
    lm_I.fill(0)  
    Rt *= np.exp(nu_beta * np.random.normal(0, 1, size=Rt.shape))
    for ti in range(t):
        kernel_values = omega_I * (1 - omega_I) ** (t - ti - 1)
        lm_I += observed_data['obs'][ti] * kernel_values  # Broadcasting for all particles
    # Adjust lambda with  cumulative infections
    lm_I = np.random.poisson((1 - C_I / N) * Rt * lm_I , size=lm_I.shape)
    
    # Update cumulative infections
    C_I += lm_I
    # Stack updated variables
    updated_state = np.column_stack((lm_I, C_I, Rt))
    # Return updated state as DataFrame
    return pd.DataFrame(updated_state, columns=state_names)

