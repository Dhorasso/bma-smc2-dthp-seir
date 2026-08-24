"""
BMA-SMC^2 applied to the Irish COVID-19 epidemic (HPSC detailed statistics).

Expects `data/real/COVID-19_HPSC_Detailed_Statistics_Profile.csv` to be
present -- see data/real/README.md.

Usage (from the repo root):
    python -m real_data_study.covid_application
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.epidemic_models import stochastic_seir_model, dthp_model
from src.observation_models import obs_dist_negative_binomial
from src.scenario_runner import analyse_replicate

SCENARIO_NAME = "Irish COVID-19"
DATA_PATH = "data/real/COVID-19_HPSC_Detailed_Statistics_Profile.csv"
FIGURES_DIR = "figures/real/covid"
CACHE_PATH = "results/real/covid/results.pkl"

N_POP = 5.16e6
E0, I0 = 5, 15
FDAY = 21  # forecast horizon (days held out)
NUM_STATE_PARTICLES = 400
NUM_THETA_PARTICLES = 800
N_DAYS = 293  # number of days of data to use, starting from day 0


def load_data():
    df = pd.read_csv(DATA_PATH)
    data = df.iloc[0:N_DAYS].copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data['obs'] = data['ConfirmedCovidCases']
    return data


def plot_raw_data(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data['Date'], data['ConfirmedCovidCases'], color='blue')
    # ax.axvline(x=separation_point, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of Cases", fontsize=12)
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# --- Priors -----------------------------------------------------------------

STATE_INFO_SEIRS = {
    'S': {'prior': [N_POP - E0 - I0, N_POP, 0, 0, 'uniform']},
    'E': {'prior': [E0, E0, 0, 0, 'uniform']},
    'I': {'prior': [0, I0, 0, 0, 'uniform']},
    'R': {'prior': [0, 0, 0, 0, 'uniform']},
    'NI': {'prior': [0, 0, 0, 0, 'uniform']},
    'Rt': {'prior': [0, np.inf, 3.2, 0.05, 'normal']},
}

THETA_INFO_SEIRS = {
    'sigma': {'prior': [1 / 5, 1 / 3, 1 / 4, 0.1, 'truncnorm', 'log']},
    'gamma': {'prior': [1 / 7.5, 1 / 4.5, 1 / 6, 0.2, 'truncnorm', 'log']},
    'nu_beta': {'prior': [0.05, 0.15, 0.1, 0.05, 'uniform', 'log']},
    'phi': {'prior': [1e-5, 0.2, 0, 0, 'uniform', 'log']},
}

STATE_INFO_DTHP = {
    'NI': {'prior': [0, I0, 0, 0, 'uniform']},
    'Rt': {'prior': [0, np.inf, 3.2, 0.05, 'normal']},
}

THETA_INFO_DTHP = {
    'omega_I': {'prior': [0, 1, 0, 0, 'uniform', 'log']},
    'nu_beta': {'prior': [0.05, 0.15, 0.1, 0.05, 'truncnorm', 'log']},
    'phi': {'prior': [1e-5, 0.2, 0, 0, 'uniform', 'log']},
}

PARAM_LABELS = {
    'omega_I': r'$\omega$', 'nu_beta': r'$\nu$', 'phi': r'$\phi$',
    'sigma': r'$\sigma$', 'gamma': r'$\gamma$',
}


def main():
    data = load_data()
    separation_point = data['Date'].iloc[-FDAY]
    plot_raw_data(data, separation_point)

    days = len(data) - FDAY

    # No ground truth for real data, so `true_values` is left empty.
    return analyse_replicate(
        scenario_name=SCENARIO_NAME,
        model_seir=stochastic_seir_model,
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
        time_col="Date",
        date_axis=True,
        cache_path=CACHE_PATH,
    )


if __name__ == "__main__":
    main()
