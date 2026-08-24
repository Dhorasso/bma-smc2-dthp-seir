###########################################################################
#  This file contains the code for different (SEIR, SEIRS and DTHP)
#######################################################################

import pandas as pd
import numpy as np
from numpy.random import binomial, normal 


######################################################################################################
#####  SEIR model with time-varying Beta as a geometric random walk #################################

def stochastic_seir_model(y, theta, theta_names, dt=1):
    # Unpack compartments
    S, E, I, R, NI, Rt = y.T
    N = S + E + I + R

    # Parameters
    param = dict(zip(theta_names, theta))
    gamma = param['gamma']
    sigma = param['sigma']
    nu_beta = param['nu_beta']  # now noise on Rt

    # Compute beta from Rt
    B = gamma * Rt

    # Transition probabilities
    P_SI = 1 - np.exp(-B * I / N * dt)
    P_EI = 1 - np.exp(-sigma * dt)
    P_IR = 1 - np.exp(-gamma * dt)

    # Transitions
    Y_SE = np.random.binomial(S.astype(int), P_SI)
    Y_EI = np.random.binomial(E.astype(int), P_EI)
    Y_IR = np.random.binomial(I.astype(int), P_IR)

    # Update compartments
    S_next = S - Y_SE
    E_next = E + Y_SE - Y_EI
    I_next = I + Y_EI - Y_IR
    R_next = R + Y_IR

    # Update Rt AFTER state update
    Rt_next = Rt * np.exp(nu_beta * np.random.normal(0, 1, size=Rt.shape) * dt)

    NI_next = Y_EI

    y_next = np.column_stack((S_next, E_next, I_next, R_next, NI_next, Rt_next))
    return np.maximum(y_next, 0)


######################################################################################################
#####  SEIR model with time-varying Beta as a geometric random walk #################################

def stochastic_seirs_model(y, theta, theta_names, dt=1):
    S, E, I, R, NI, Rt = y.T
    N = S + E + I + R

    param = dict(zip(theta_names, theta))
    gamma = param['gamma']
    sigma = param['sigma']
    alpha = param['alpha']
    nu_beta = param['nu_beta']

    tau = 1 / (80 * 52)

    # beta from Rt
    B = Rt*(sigma + tau )*(gamma +tau )/sigma

    # Probabilities
    P_SE = 1 - np.exp(-B * I / N * dt)
    P_EI = 1 - np.exp(-sigma * dt)
    P_IR = 1 - np.exp(-gamma * dt)
    P_RS = 1 - np.exp(-alpha * dt)

    # Transitions
    Y_SE = np.random.binomial(S.astype(int), P_SE)
    Y_EI = np.random.binomial(E.astype(int), P_EI)
    Y_IR = np.random.binomial(I.astype(int), P_IR)
    Y_RS = np.random.binomial(R.astype(int), P_RS)

    # Updates
    S_next = S - Y_SE + Y_RS +tau * (N - S) * dt
    E_next = E + Y_SE - Y_EI -tau * E * dt
    I_next = I + Y_EI - Y_IR -tau * I * dt
    R_next = R + Y_IR - Y_RS -tau * R * dt

    # Update Rt LAST
    Rt_next = Rt * np.exp(nu_beta * np.random.normal(0, 1, size=Rt.shape) * dt)

    NI_next = Y_EI

    y_next = np.column_stack((S_next, E_next, I_next, R_next, NI_next, Rt_next))
    return np.maximum(y_next, 0)



####################################################################
######### Discrete-time Hawkes Process    ########################

def dthp_model(state, theta, state_names, theta_names, observed_data, t, N):
    lm_I, Rt = state.T.copy()

    param = dict(zip(theta_names, theta))
    omega_I = param['omega_I']
    nu_beta = param['nu_beta']

    Rt *= np.exp(nu_beta * np.random.normal(0, 1, size=Rt.shape))

    lm_I.fill(0)

    # -------- NORMALIZED KERNEL --------
    weights = []
    if t==0 :
         weights.append(1)
    else :
        for ti in range(t):
            k = t - ti -1
            weights.append(omega_I * (1 - omega_I) ** k)

    weights = np.array(weights)
    # weights = weights / weights.sum()  # normalization

    # Compute lambda
    for ti in range(t):
        lm_I += observed_data['obs'].iloc[ti] * weights[ti]

    cum_obs = observed_data['obs'].iloc[:max(1, t-1)].sum()

    lm_I *= (1 - cum_obs / N) * Rt

    updated_state = np.column_stack((lm_I, Rt))
    return pd.DataFrame(updated_state, columns=state_names)