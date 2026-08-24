"""
Scenario A: smooth (sigmoid) transition in transmission, "truth" generated
from the SEIR model. Data-generating process, priors, and figures match the
Scenario A results in the paper.

Usage (from the repo root):
    python -m simulation_study.scenario_A
"""

import os

import numpy as np

from src.epidemic_models import stochastic_seir_model, dthp_model
from src.observation_models import obs_dist_negative_binomial
from src.simulate import solve_seir_var_beta, load_or_generate_replicates
from src.scenario_runner import analyse_replicate

SCENARIO_NAME = "Scenario A"
DATA_DIR = "data/simulated/scenario_A"
FIGURES_DIR = "figures/simulated/scenario_A"
CACHE_PATH = "results/simulated/scenario_A/rep0.pkl"
N_REPS = 10
BASE_SEED = 123

N_POP = 50000
TRUE_THETA = [1 / 2, 1 / 6]  # sigma (incubation rate), gamma (recovery rate)
T_START, T_END = 0, 100
FDAY = 21  # forecast horizon (days held out for out-of-sample evaluation)


def beta_func(t):
    """
    Smooth sigmoid-like transition in transmission:
    Rt ~ 0.8 for days 0-40, a rapid transition over days 40-55, then Rt ~ 3.5.
    Tests whether the model can anticipate the START of exponential growth
    when trained only on the stable period.
    """
    return np.exp(np.cos(2 * np.pi * t / 96) - t / 125) * 0.28


def simulate_one_replicate():
    initial_state = [N_POP - 10, 0, 10, 0, 0, beta_func(0)]
    return solve_seir_var_beta(TRUE_THETA, initial_state, T_START, T_END, beta_func=beta_func)


# --- Priors -----------------------------------------------------------------

STATE_INFO_SEIR = {
    'S': {'prior': [N_POP - 15, N_POP, 0, 0, 'uniform']},
    'E': {'prior': [0, 5, 0, 0, 'uniform']},
    'I': {'prior': [0, 10, 0, 0, 'uniform']},
    'R': {'prior': [0, 0, 0, 0, 'uniform']},
    'NI': {'prior': [0, 0, 0, 0, 'uniform']},
    'Rt': {'prior': [4.2, 4.5, 4.25, 0.15, 'uniform']},
}

THETA_INFO_SEIR = {
    'sigma': {'prior': [0, 1, 0.45, 0.1, 'truncnorm', 'log']},
    'gamma': {'prior': [0, 0.2, 0.15, 0.05, 'truncnorm', 'log']},
    'nu_beta': {'prior': [0.05, 0.2, 0.1, 0.01, 'truncnorm', 'log']},
    'phi': {'prior': [1e-5, 0.2, 0, 0, 'uniform', 'log']},
}

STATE_INFO_DTHP = {
    'NI': {'prior': [0, 5, 0, 0, 'uniform']},
    'Rt': {'prior': [4.2, 4.5, 4.25, 0.15, 'uniform']},
}

THETA_INFO_DTHP = {
    'omega_I': {'prior': [0, 1, 0.12, 0.06, 'truncnorm', 'log']},
    'nu_beta': {'prior': [0.05, 0.2, 0.1, 0.01, 'truncnorm', 'log']},
    'phi': {'prior': [1e-5, 0.2, 0, 0, 'uniform', 'log']},
}

PARAM_LABELS = {
    'omega_I': r'$\omega$', 'nu_beta': r'$\nu$', 'phi': r'$\phi$',
    'sigma': r'$\sigma$', 'gamma': r'$\gamma$',
}
TRUE_VALUES = {'sigma': TRUE_THETA[0], 'gamma': TRUE_THETA[1]}


def main():
    replicates = load_or_generate_replicates(
        simulate_one_replicate, n_reps=N_REPS, base_seed=BASE_SEED, out_dir=DATA_DIR,
    )

    # The analysis below runs on a single replicate (index 0). To inspect a
    # different replicate, change `replicate_idx`.
    replicate_idx = 0
    simulated_data = replicates[replicate_idx]
    days = len(simulated_data) - FDAY

    return analyse_replicate(
        scenario_name=SCENARIO_NAME,
        model_seir=stochastic_seir_model,
        model_dthp=dthp_model,
        state_info_seir=STATE_INFO_SEIR,
        theta_info_seir=THETA_INFO_SEIR,
        state_info_dthp=STATE_INFO_DTHP,
        theta_info_dthp=THETA_INFO_DTHP,
        observed_data=simulated_data[:days],
        forecast_days=FDAY,
        N_pop=N_POP,
        observation_distribution=obs_dist_negative_binomial,
        figures_dir=FIGURES_DIR,
        param_labels=PARAM_LABELS,
        true_values=TRUE_VALUES,
        full_data=simulated_data,
        time_col="time",
        true_rt_col="Rt",
        cache_path=CACHE_PATH,
        # style={"color_dthp": "crimson", "ylim_weight": (0, 1)},  # customise without re-running SMC^2
    )


if __name__ == "__main__":
    main()
