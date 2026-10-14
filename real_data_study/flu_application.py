"""
BMA-SMC^2 applied to the Irish 2024 influenza epidemic.

Expects `data/real/Influenza2024.xlsx` (weekly influenza case counts) to be
present -- see data/real/README.md.

Usage (from the repo root):
    python -m real_data_study.flu_application
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.epidemic_models import stochastic_seirs_model, dthp_model
from src.observation_models import obs_dist_negative_binomial
from src.scenario_runner import analyse_replicate

SCENARIO_NAME = "Irish Influenza 2024"
DATA_PATH = "data/real/Influenza2024.xlsx"
FIGURES_DIR = "figures/real/flu"
CACHE_PATH = "results/real/flu/results.pkl"

N_POP = 5.16e6
E0, I0 = 5, 15
FDAY = 14  # forecast horizon (weeks held out)
NUM_STATE_PARTICLES = 400
NUM_THETA_PARTICLES = 800


def load_data():
    df = pd.read_excel(DATA_PATH)
    # Restrict to observations from 29 May 2022 onward
    data = df.iloc[72:-2].copy().reset_index(drop=True)
    data['Week ending'] = pd.to_datetime(data['Week ending'])
    data['obs'] = data['Number of influenza cases']
    return data


def plot_raw_data(data):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(data['Week ending'], data['Number of influenza cases'], color='blue')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Weekly Cases", fontsize=12)
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# --- Priors -----------------------------------------------------------------

STATE_INFO_SEIRS = {
    'S': {'prior': [N_POP - E0 - I0, N_POP, 0, 0, 'uniform']},
    'E': {'prior': [0, E0, 0, 0, 'uniform']},
    'I': {'prior': [0, I0, 0, 0, 'uniform']},
    'R': {'prior': [0, 0, 0, 0, 'uniform']},
    'NI': {'prior': [0, 0, 0, 0, 'uniform']},
    'Rt': {'prior': [0, np.inf, 0.5, 0.05, 'normal']},
}

THETA_INFO_SEIRS = {
    'sigma': {'prior': [7 / 3, 7, 7 / 1.5, 0.1, 'truncnorm', 'log']},
    'gamma': {'prior': [7 / 7, 7 / 5, 7 / 6, 0.1, 'truncnorm', 'log']},
    'alpha': {'prior': [1 / (6 * 4), 1 / (3 * 4), 1 / 2, 0.1, 'uniform', 'log']},
    'nu_beta': {'prior': [0.05, 0.15, 0.1, 0.02, 'uniform', 'log']},
    'phi': {'prior': [1e-5, 0.2, 0, 0, 'uniform', 'log']},
}

STATE_INFO_DTHP = {
    'NI': {'prior': [0, I0, 0, 0, 'uniform']},
    'Rt': {'prior': [0, np.inf, 0.5, 0.05, 'normal']},
}

THETA_INFO_DTHP = {
    'omega_I': {'prior': [0, 1, 0, 0, 'uniform', 'log']},
    'nu_beta': {'prior': [0.05, 0.15, 0.1, 0.02, 'uniform', 'log']},
    'phi': {'prior': [1e-5, 0.2, 0, 0, 'uniform', 'log']},
}

PARAM_LABELS = {
    'omega_I': r'$\omega$', 'nu_beta': r'$\nu$', 'phi': r'$\phi$',
    'sigma': r'$\sigma$', 'gamma': r'$\gamma$', 'alpha': r'$\alpha$',
}


def main():
    data = load_data()
    plot_raw_data(data)

    days = len(data) - FDAY

    # No ground truth for real data, so `true_values` is left empty.
    return analyse_replicate(
        scenario_name=SCENARIO_NAME,
        model_seir=stochastic_seirs_model,
        model_dthp=dthp_model,
        state_info_seir=STATE_INFO_SEIRS,
        theta_info_seir=THETA_INFO_SEIRS,
        state_info_dthp=STATE_INFO_DTHP,
        theta_info_dthp=THETA_INFO_DTHP,
        observed_data=data[:days],
        forecast_days=FDAY,
        N_pop=N_POP,
        observation_distribution=obs_dist_negative_binomial,
        figures_dir=FIGURES_DIR,
        param_labels=PARAM_LABELS,
        num_state_particles=NUM_STATE_PARTICLES,
        num_theta_particles=NUM_THETA_PARTICLES,
        full_data=data,
        time_col="Week ending",
        date_axis=True,
        cache_path=CACHE_PATH,
    )


if __name__ == "__main__":
    main()
