###########################################################################
#  This file contains the code for different (SEIR, SEIRS and DTHP)
#######################################################################

import pandas as pd
import numpy as np
from numpy.random import binomial, normal 


######################################################################################################
#####  SEIR model with time-varying Beta as a geometric random walk #################################

def stochastic_seir_model(y, theta, theta_names, dt=1):
    """
    Vectorized discrete-time stochastic SEIR model with time-varying Beta (transmission rate).
    
    Parameters:
    ----------
    y : np.ndarray
        A 2D array with shape (num_particles, num_compartments). Columns:
        [S, E, I, R, NI, B]
    theta : np.ndarray
        A 1D array of parameter values for each particle.
    theta_names : list of str
        List of parameter names matching the order in `theta`.
    dt : float
        Time step size.

    Returns:
    -------
    np.ndarray
        Updated compartment array with same shape as input `y`.
    """
    # Unpack compartments
    S, E, I, R, NI, B = y.T
    N = S + E + I + R

    # Parameters
    param = dict(zip(theta_names, theta))
    gamma = param['gamma']       # Recovery rate
    sigma = param['sigma']       # Incubation rate (E → I)
    nu_beta = param['nu_beta']   # Volatility in transmission rate

    # Transition probabilities
    P_SI = 1 - np.exp(-B * I / N * dt)    # S → E
    P_EI = 1 - np.exp(-sigma * dt)        # E → I
    P_IR = 1 - np.exp(-gamma * dt)        # I → R

    # Binomial transitions
    Y_SE = np.random.binomial(S.astype(int), P_SI)
    Y_EI = np.random.binomial(E.astype(int), P_EI)
    Y_IR = np.random.binomial(I.astype(int), P_IR)

    # Update compartments
    S_next = S - Y_SE
    E_next = E + Y_SE - Y_EI
    I_next = I + Y_EI - Y_IR
    R_next = R + Y_IR

    # Update β with geometric random walk
    B_next = B * np.exp(nu_beta * np.random.normal(0, 1, size=B.shape) * dt)

    # New infections = S → E
    NI_next = Y_EI

    y_next = np.column_stack((S_next, E_next, I_next, R_next, NI_next, B_next))
    return np.maximum(y_next, 0)



######################################################################################################
#####  SEIR model with time-varying Beta as a geometric random walk #################################

def stochastic_seirs_model(y, theta, theta_names, dt=1):
    """
    Vectorized discrete-time stochastic SEIRS model with time-varying Beta.

    Parameters:
    ----------
    y : np.ndarray
        A 2D array of compartments:
        [S, E, I, R, NI, B]
    theta : np.ndarray
        Parameter values per particle.
    theta_names : list of str
        Parameter names.
    dt : float
        Time step.

    Returns:
    -------
    np.ndarray
        Updated compartments.
    """
    # Unpack compartments
    S, E, I, R, NI, B = y.T
    N = S + E + I + R

    # Parameters
    param = dict(zip(theta_names, theta))
    gamma = param['gamma']       # I → R
    sigma = param['sigma']       # E → I
    alpha = param['alpha']       # R → S (waning immunity)
    nu_beta = param['nu_beta']   # volatility

    mu = 1 / (80 * 52)  # optional natural death/birth rate (approx weekly)

    # Transition probabilities
    P_SE = 1 - np.exp(-B * I / N * dt)
    P_EI = 1 - np.exp(-sigma * dt)
    P_IR = 1 - np.exp(-gamma * dt)
    P_RS = 1 - np.exp(-alpha * dt)

    # Binomial transitions
    Y_SE = np.random.binomial(S.astype(int), P_SE)
    Y_EI = np.random.binomial(E.astype(int), P_EI)
    Y_IR = np.random.binomial(I.astype(int), P_IR)
    Y_RS = np.random.binomial(R.astype(int), P_RS)

    # Compartment updates with optional demographics
    S_next = S - Y_SE + Y_RS + mu * (N - S) * dt
    E_next = E + Y_SE - Y_EI - mu * E * dt
    I_next = I + Y_EI - Y_IR - mu * I * dt
    R_next = R + Y_IR - Y_RS - mu * R * dt

    B_next = B * np.exp(nu_beta * np.random.normal(0, 1, size=B.shape) * dt)
    NI_next = Y_EI

    y_next = np.column_stack((S_next, E_next, I_next, R_next, NI_next, B_next))
    return np.maximum(y_next, 0)



####################################################################
######### Discrete-time Hawkes Process    ########################

def dthp_model(state, theta, state_names, theta_names, observed_data, t, N):
    """
    Vectorized Discrete-time Hawkes Process for flu modeling.

    Parameters:
    - state: 2D array of state variables (num_particles x num_state_variables)
    - theta: 2D array of parameters (num_particles x num_parameters)
    - state_names: List of state variable names
    - theta_names: List of parameter names
    - observed_data: Dictionary containing observed data, including 'obs' key with array (num_time_steps)
    - t: Current time step (scalar)
    - N : Total population size

    Returns:
    - updated_state: DataFrame of updated state variables with state names as columns
    """

    # Unpack state variables
    lm_I, Rt = state.T.copy()  # Ensure state variables are writable

    # Create parameter dictionary
    param = dict(zip(theta_names, theta))
    omega_I = param['omega_I']
    nu_beta = param.get('nu_beta', 0.1)

    # Reset lambdas
    lm_I.fill(0)

    # Update Rt with log-normal noise
    Rt *= np.exp(nu_beta * np.random.normal(0, 1, size=Rt.shape))
    
    # Compute lambda using past observed infections
    for ti in range(t): # it sport at t-1#
        kernel_values = omega_I * (1 - omega_I) ** (t - ti - 1)
        # ti=min(ti, len(observed_data)-1)
        lm_I += observed_data['obs'].iloc[ti] * kernel_values

    cum_obs = observed_data['obs'].iloc[:max(1,t-1)].sum()

    lm_I *= (1 - cum_obs / N)* Rt 
    # new_infections = np.random.poisson(mean_infections)

    # Update state
    updated_state = np.column_stack((lm_I, Rt))
    return pd.DataFrame(updated_state, columns=state_names)


