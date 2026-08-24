"""
Scenario C: "truth" generated from the DTHP model instead of the SEIR model
(tests how well each model, and the ensemble, recovers a DTHP-generated
epidemic). Data-generating process, priors, and figures match the
Scenario C results in the paper.

Usage (from the repo root):
    python -m simulation_study.scenario_C
"""

from src.epidemic_models import stochastic_seir_model, dthp_model
from src.observation_models import obs_dist_negative_binomial
from src.simulate import dthp_var_rt_simulation, load_or_generate_replicates
from src.scenario_runner import analyse_replicate

SCENARIO_NAME = "Scenario C"
DATA_DIR = "data/simulated/scenario_C"
FIGURES_DIR = "figures/simulated/scenario_C"
CACHE_PATH = "results/simulated/scenario_C/rep0.pkl"
N_REPS = 10
BASE_SEED = 123

N_POP = 50000
TRUE_THETA_DTHP = [0.2]  # omega: geometric triggering-kernel decay
INITIAL_STATE_DTHP = [15, 1.5]  # [NI_0, Rt_0]
T_START, T_END = 0, 100
FDAY = 21


def piecewise_Rt(t):
    """Piecewise-linear reproduction number used to generate the DTHP truth."""
    if t < 35:
        return 1.5
    elif t < 50:
        return 1.5 - ((1.5 - 0.9) / (50 - 35)) * (t - 35)
    else:
        return 0.9 + ((0.93 - 0.9) / (100 - 50)) * (t - 50)


def simulate_one_replicate():
    return dthp_var_rt_simulation(
        TRUE_THETA_DTHP, INITIAL_STATE_DTHP, T_START, T_END,
        Rt_func=piecewise_Rt, N_pop=N_POP,
    )


# --- Priors (both SEIR and DTHP are still *fit* to the DTHP-generated data) -

STATE_INFO_SEIR = {
    'S': {'prior': [N_POP - 15, N_POP, 0, 0, 'uniform']},
    'E': {'prior': [0, 5, 0, 0, 'uniform']},
    'I': {'prior': [0, 10, 0, 0, 'uniform']},
    'R': {'prior': [0, 0, 0, 0, 'uniform']},
    'NI': {'prior': [0, 0, 0, 0, 'uniform']},
    'Rt': {'prior': [1.4, 1.5, 1.4, 0.05, 'uniform']},
}

THETA_INFO_SEIR = {
    'sigma': {'prior': [0, 0.5, 0.5, 0.1, 'truncnorm', 'log']},
    'gamma': {'prior': [0, 0.3, 0.5, 0.5, 'uniform', 'log']},
    'nu_beta': {'prior': [0.05, 0.2, 0.1, 0.01, 'truncnorm', 'log']},
    'phi': {'prior': [1e-5, 0.2, 0, 0, 'uniform', 'log']},
}

STATE_INFO_DTHP = {
    'NI': {'prior': [0, 5, 0, 0, 'uniform']},
    'Rt': {'prior': [1.4, 1.5, 1.4, 0.05, 'uniform']},
}

THETA_INFO_DTHP = {
    'omega_I': {'prior': [0, 1, 0.2, 0.2, 'truncnorm', 'log']},
    'nu_beta': {'prior': [0.05, 0.2, 0.1, 0.01, 'truncnorm', 'log']},
    'phi': {'prior': [1e-5, 0.2, 0, 0, 'uniform', 'log']},
}

PARAM_LABELS = {
    'omega_I': r'$\omega$', 'nu_beta': r'$\nu$', 'phi': r'$\phi$',
    'sigma': r'$\sigma$', 'gamma': r'$\gamma$',
}
# Ground truth only exists for the DTHP triggering-kernel parameter here,
# since the data-generating process is DTHP rather than SEIR.
TRUE_VALUES = {'omega_I': TRUE_THETA_DTHP[0]}


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
