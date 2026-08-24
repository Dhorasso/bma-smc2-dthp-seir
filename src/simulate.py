"""
Data-generating processes used by the simulation study, plus helpers to
generate a batch of independent replicates and save/load them as CSV so the
(slow) generation step only has to run once.

Two generators are provided:
- `solve_seir_var_beta` : SEIR-type model with a user-supplied time-varying
  transmission rate `beta_func(t)` (used for scenarios A and B).
- `dthp_var_rt_simulation` : discrete-time Hawkes process with a
  user-supplied time-varying reproduction number `Rt_func(t)` (used for
  scenario C, where the "truth" is DTHP rather than SEIR).
"""

import os

import numpy as np
import pandas as pd
from numpy.random import binomial


# ---------------------------------------------------------------------------
# SEIR with a custom time-varying transmission rate
# ---------------------------------------------------------------------------

def seir_var_beta(y, theta, t, dt=1, beta_func=None):
    """One-step SEIR transition with beta(t) supplied by `beta_func`."""
    S, E, I, R, NI, B = y
    N = S + E + I + R

    sigma, gamma = theta[0], theta[1]
    B = beta_func(t)

    P_SE = 1 - np.exp(-B * I / N * dt)
    P_EI = 1 - np.exp(-sigma * dt)
    P_IR = 1 - np.exp(-gamma * dt)

    B_SE = binomial(S, P_SE)
    B_EI = binomial(E, P_EI)
    B_IR = binomial(I, P_IR)

    S -= B_SE
    E += B_SE - B_EI
    I += B_EI - B_IR
    R += B_IR
    NI = B_EI

    return [max(0, val) for val in [S, E, I, R, NI, B]]


def solve_seir_var_beta(theta, initial_state, t_start, t_end, dt=1, beta_func=None):
    """
    Simulate an SEIR trajectory with time-varying beta(t).

    Returns a DataFrame with columns S, E, I, R, NI, B, time, obs, Rt
    ('obs' = NI, 'Rt' = B / gamma).
    """
    t_values = np.arange(t_start, t_end + dt, dt)
    results = np.zeros((len(t_values), len(initial_state)))
    results[0] = initial_state

    for i in range(1, len(t_values)):
        results[i] = seir_var_beta(results[i - 1], theta, i, dt, beta_func=beta_func)

    results_df = pd.DataFrame(results, columns=['S', 'E', 'I', 'R', 'NI', 'B'])
    results_df['time'] = t_values
    results_df['obs'] = results_df['NI']
    results_df['Rt'] = results_df['B'] / theta[1]
    return results_df


# ---------------------------------------------------------------------------
# Discrete-time Hawkes process with a custom time-varying Rt
# ---------------------------------------------------------------------------

def dthp_var_rt_simulation(theta, initial_state, t_start, t_end, Rt_func, dt=1, N_pop=50000):
    """
    Simulate a discrete-time Hawkes process trajectory with Rt(t) supplied
    by `Rt_func`.

    Parameters
    ----------
    theta : [omega] geometric triggering-kernel decay parameter.
    initial_state : [NI_0, Rt_0].
    Rt_func : callable, Rt_func(t) -> reproduction number at time t.
    N_pop : total population size (used for the susceptible-depletion factor).

    Returns a DataFrame with columns time, NI, Rt, obs (= NI).
    """
    omega = theta[0]
    t_values = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t_values)

    NI = np.zeros(n_steps)
    Rt = np.zeros(n_steps)
    C_I = np.zeros(n_steps)

    NI[0] = initial_state[0]
    Rt[0] = initial_state[1]
    C_I[0] = NI[0]

    def phi(delay, omega):
        """Geometric triggering kernel."""
        return omega * (1 - omega) ** (delay - 1)

    for t in range(1, n_steps):
        Rt[t] = Rt_func(t)
        susceptible_fraction = max(0, 1 - C_I[t - 1] / N_pop)

        excitation = sum(NI[s] * phi(t - s, omega) for s in range(t))
        lambda_t = max(0, susceptible_fraction * Rt[t] * excitation)

        NI[t] = np.random.poisson(lambda_t)
        C_I[t] = C_I[t - 1] + NI[t]

    return pd.DataFrame({'time': t_values, 'NI': NI, 'Rt': Rt, 'obs': NI})


# ---------------------------------------------------------------------------
# Replicate generation / caching to CSV
# ---------------------------------------------------------------------------

def generate_replicates(simulate_fn, n_reps, base_seed, out_dir, **simulate_kwargs):
    """
    Run `simulate_fn(**simulate_kwargs)` `n_reps` times (seeded
    `base_seed + rep`), saving each replicate to `out_dir/rep_{i}.csv`.

    Returns the list of generated DataFrames.
    """
    os.makedirs(out_dir, exist_ok=True)
    replicates = []

    for rep in range(n_reps):
        np.random.seed(base_seed + rep)
        df = simulate_fn(**simulate_kwargs)
        df.to_csv(os.path.join(out_dir, f"rep_{rep}.csv"), index=False)
        replicates.append(df)
        print(f"  saved {out_dir}/rep_{rep}.csv")

    return replicates


def load_replicates(out_dir, n_reps):
    """Load `rep_0.csv .. rep_{n_reps-1}.csv` from `out_dir`."""
    return [
        pd.read_csv(os.path.join(out_dir, f"rep_{rep}.csv"))
        for rep in range(n_reps)
    ]


def load_or_generate_replicates(simulate_fn, n_reps, base_seed, out_dir, **simulate_kwargs):
    """Load replicates from `out_dir` if they already exist, else generate + save them."""
    expected_files = [os.path.join(out_dir, f"rep_{rep}.csv") for rep in range(n_reps)]
    if all(os.path.exists(f) for f in expected_files):
        print(f"Loading {n_reps} cached replicates from {out_dir}/")
        return load_replicates(out_dir, n_reps)

    print(f"Generating {n_reps} replicates into {out_dir}/ ...")
    return generate_replicates(simulate_fn, n_reps, base_seed, out_dir, **simulate_kwargs)
