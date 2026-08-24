
######################################################################################################
#####  Fonction to propagate the state forward  ######################################################
import numpy as np
import pandas as pd

def state_transition(model, theta, initial_state, state_names, theta_names, t_start, t_end, dt=1):
    """
    Solve a stochastic disease model using vectorized computation.

    Parameters:
    - model: Model function
    - theta: Parameter array
    - initial_state: Initial state array (num_particles x num_compartments)
    - state_names: Names of compartments
    - theta_names: Names of parameters
    - t_start: Start time
    - t_end: End time
    - dt: Time step (default is 1)

    Returns:
    - results_df: DataFrame containing results at the last time step
    """
    # Time points
    t_points = np.arange(t_start, t_end + dt, dt)
    num_steps = len(t_points)

    # Initialize results array
    num_particles, num_compartments = initial_state.shape
    results = np.zeros((num_steps, num_particles, num_compartments))
    results[0] = initial_state  # Set initial state

    # Run the model for each time step
    for i in range(1, num_steps):
        results[i] = model(results[i - 1], theta, theta_names, dt)

    # Return results as DataFrame for the last time step
    return pd.DataFrame(results[-1], columns=state_names)

