###############################################################################################################
#  This file contains the code for different types of stochastic SEIR models. This can be extended by the user
#################################################################################################################

import pandas as pd
import numpy as np
from numpy.random import binomial, normal 

######################################################################################################
#####  SEAIR model with time-varying Beta as a geometric random walk #################################

def stochastic_seair_model(y, theta, theta_names, dt=1):
    """
    Vectorized discrete-time stochastic SEAIR compartmental model.
    
    Parameters:
    ----------
    y : np.ndarray
        A 2D array of compartments with shape (num_particles, num_compartments). 
        Columns represent the following compartments:
        [S (Susceptible), E (Exposed), A (Asymptomatic), I (Infected), R (Recovered), 
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
    S, E, A, I, R, NI, B = y.T

    # Calculate total population for each particle
    N = S + E + A + I + R

    # Unpack parameters into a dictionary for easy access
    param = dict(zip(theta_names, theta))
    pa = param.get('pa', 0.5)  # Default value for fraction asymptomatic if not specified
    ra = param.get('ra', 0.5)  # Default value for  reduction transmission asymptomatic if not specified
    sigma = param['sigma']     # latency rate
    gamma = param['gamma']     # recovery rate   
    nu_beta = param.get('nu_beta', 0.1)  # Default value for `nu_beta` if not specified

    # Transition probabilities (vectorized)
    P_SE = 1 - np.exp(-B * (I + ra * A) / N * dt)  # Susceptible → Exposed
    P_EAI = 1 - np.exp(-sigma * dt)                 # Exposed → Asymptomatic/Infected
    P_AR = 1 - np.exp(-gamma * dt)                  # Asymptomatic → Recovered
    P_IR = P_AR                                      # Infected → Recovered

    # Simulate transitions using binomial draws
    Y_SE = binomial(S.astype(int), P_SE)   # S → E
    Y_EAI = binomial(E.astype(int), P_EAI)  # E → A/I
    Y_EA = binomial(Y_EAI, pa)              # E → A
    Y_EI = Y_EAI - Y_EA                     # E → I
    Y_AR = binomial(A.astype(int), P_AR)    # A → R
    Y_IR = binomial(I.astype(int), P_IR)    # I → R

    # Update compartments
    S_next = S - Y_SE
    E_next = E + Y_SE - Y_EAI
    A_next = A + Y_EA - Y_AR
    I_next = I + Y_EI - Y_IR
    R_next = R + Y_AR + Y_IR
    
    # Update transmission rate with stochastic volatility
    B_next = B * np.exp(nu_beta * normal(0, 1, size=B.shape) * dt)

    # Update new infected
    NI_next = Y_EI

    # Combine updated compartments into a 2D array
    y_next = np.column_stack((S_next, E_next, A_next, I_next, R_next, NI_next, B_next))
    
    # Ensure all compartments remain non-negative
    return np.maximum(y_next, 0)

######################################################################################################
#####  SEIR model with time-varying Beta as a geometric random walk #################################

def stochastic_seir_model(y, theta, theta_names, dt=1):
    """
    Vectorized discrete-time stochastic SEIR compartmental model.
    Parameters:
    ----------
    y : np.ndarray
        A 2D array of compartments with shape (num_particles, num_compartments). 
        Columns represent the following compartments:
        [S (Susceptible), E (Exposed), I (Infected), R (Recovered), 
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
    S, E, I, R, NI, B = y.T

    # Calculate total population for each particle
    N = S + E + I + R

    # Unpack parameters into a dictionary for easy access
    param = dict(zip(theta_names, theta))
    sigma = param['sigma']     # latency rate
    gamma = param['gamma']     # recovery rate   
    nu_beta = param.get('nu_beta', 0.1)  # Default value for `nu_beta` if not specified

    # Transition probabilities (vectorized)
    P_SE = 1 - np.exp(-B * I / N * dt)             # Susceptible → Exposed
    P_EI = 1 - np.exp(-sigma * dt)                 # Exposed → Infected
    P_IR = 1 - np.exp(-gamma * dt)                  # Infected → Recovered

    # Simulate transitions using binomial draws
    Y_SE = binomial(S.astype(int), P_SE)   # S → E
    Y_EI = binomial(E.astype(int), P_EI)  # E → I
    Y_IR = binomial(I.astype(int), P_IR)    # I → R

    # Update compartments
    S_next = S - Y_SE
    E_next = E + Y_SE - Y_EI
    I_next = I + Y_EI - Y_IR
    R_next = R + Y_IR
    
    # Update transmission rate with stochastic volatility
    B_next = B * np.exp(nu_beta * normal(0, 1, size=B.shape) * dt)

    # Update new infected
    NI_next = Y_EI

    # Combine updated compartments into a 2D array
    y_next = np.column_stack((S_next, E_next, I_next, R_next, NI_next, B_next))
    
    # Ensure all compartments remain non-negative
    return np.maximum(y_next, 0)

######################################################################################################
#####  SEIR model with constant beta ######################################################

def stochastic_seir_model_const_beta(y, theta, theta_names, dt=1):
    """
    Vectorized discrete-time stochastic SEIR compartmental model.
    Parameters:
    ----------
    y : np.ndarray
        A 2D array of compartments with shape (num_particles, num_compartments). 
        Columns represent the following compartments:
        [S (Susceptible), E (Exposed), I (Infected), R (Recovered), 
         NI (new infected)].
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
    S, E, I, R, NI = y.T

    # Calculate total population for each particle
    N = S + E + I + R

    # Unpack parameters into a dictionary for easy access
    param = dict(zip(theta_names, theta))
    beta = param['beta']       # transmission rate   
    sigma = param['sigma']     # latency rate
    gamma = param['gamma']     # recovery rate   

    # Transition probabilities (vectorized)
    P_SE = 1 - np.exp(-beta * I / N * dt)             # Susceptible → Exposed
    P_EI = 1 - np.exp(-sigma * dt)                 # Exposed → Infected
    P_IR = 1 - np.exp(-gamma * dt)                  # Infected → Recovered

    # Simulate transitions using binomial draws
    Y_SE = binomial(S.astype(int), P_SE)   # S → E
    Y_EI = binomial(E.astype(int), P_EI)  # E → I
    Y_IR = binomial(I.astype(int), P_IR)    # I → R

    # Update compartments
    S_next = S - Y_SE
    E_next = E + Y_SE - Y_EI
    I_next = I + Y_EI - Y_IR
    R_next = R + Y_IR

    # Update new infected
    NI_next = Y_EI

    # Combine updated compartments into a 2D array
    y_next = np.column_stack((S_next, E_next, I_next, R_next, NI_next))
    
    # Ensure all compartments remain non-negative
    return np.maximum(y_next, 0)
