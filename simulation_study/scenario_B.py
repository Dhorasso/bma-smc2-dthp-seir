"""
Scenario B: abrupt, then reversing, step-change in transmission ("truth"
generated from the SEIR model). Data-generating process, priors, and figures
match the Scenario B results in the paper.

Usage (from the repo root):
    python -m simulation_study.scenario_B
"""

from src.epidemic_models import stochastic_seir_model, dthp_model
from src.observation_models import obs_dist_negative_binomial
from src.simulate import solve_seir_var_beta, load_or_generate_replicates
from src.scenario_runner import analyse_replicate

SCENARIO_NAME = "Scenario B"
DATA_DIR = "data/simulated/scenario_B"
FIGURES_DIR = "figures/simulated/scenario_B"
CACHE_PATH = "results/simulated/scenario_B/rep0.pkl"
N_REPS = 10
BASE_SEED = 123

N_POP = 50000
TRUE_THETA = [1 / 2, 1 / 6]  # sigma (incubation rate), gamma (recovery rate)
T_START, T_END = 0, 100
FDAY = 21


def beta_func(t):
    """
    Piecewise-linear transmission rate: stable at 0.35 until day 40, a sharp
    drop to 0.1 by day 41, stable at 0.1 until day 80, then a linear rise to
    0.29 by day 93 and stable afterwards.
    """
    if t < 40:
        return 0.35
    elif t < 41:
        return 0.35 - ((0.35 - 0.1) / (41 - 40)) * (t - 40)
    elif t < 80:
        return 0.1
    elif t < 93:
        return 0.1 + ((0.29 - 0.1) / (93 - 80)) * (t - 80)
    else:
        return 0.29


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
    'Rt': {'prior': [2, 2.5, 2, 0.05, 'uniform']},
}

THETA_INFO_SEIR = {
    'sigma': {'prior': [0, 1, 0.45, 0.1, 'truncnorm', 'log']},
    'gamma': {'prior': [0, 1, 0.1, 0.05, 'truncnorm', 'log']},
    'nu_beta': {'prior': [0.05, 0.2, 0.1, 0.01, 'truncnorm', 'log']},
    'phi': {'prior': [1e-5, 0.2, 0, 0, 'uniform', 'log']},
}

STATE_INFO_DTHP = {
    'NI': {'prior': [0, 5, 0, 0, 'uniform']},
    'Rt': {'prior': [2, 2.5, 2, 0.05, 'uniform']},
}

THETA_INFO_DTHP = {
    'omega_I': {'prior': [0, 1, 0.0, 0.05, 'truncnorm', 'log']},
    'nu_beta': {'prior': [0.05, 0.1, 0.1, 0.01, 'truncnorm', 'log']},
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
    )


if __name__ == "__main__":
    main()
