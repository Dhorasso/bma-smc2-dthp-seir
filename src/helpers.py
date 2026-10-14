"""
Helper functions shared by every scenario / application script:

- compute_window_weights : turn the two models' per-step log-evidence into a
  time-varying model weight (posterior model probability), using a sliding
  window with a burn-in guard.
- compute_model_average  : combine two models' predictive particle sets into
  one ensemble particle set using deterministic largest-remainder allocation
  according to the model weights.
- extend_array           : pad a 1D array out to a target length by repeating
  its last value (used to extend model weights into the forecast horizon).
"""

import numpy as np


def extend_array(arr, target_length):
    """Pad `arr` up to `target_length` by repeating its last value."""
    arr = np.asarray(arr)
    if len(arr) < target_length:
        return np.pad(arr, (0, target_length - len(arr)), mode='edge')
    return arr[:target_length]


def compute_window_weights(loglik_dthp, loglik_seir, window, t_burning=0):
    """
    Convert per-step log-evidence for the DTHP and SEIR/SEIRS models into a
    time-varying posterior model weight.

    A sliding window of size `window` is used to accumulate log-evidence
    (so recent predictive performance matters more than distant history),
    with a `t_burning`-step burn-in during which log-evidence accumulates
    without windowing (to avoid unstable weights at the very start).

    Parameters
    ----------
    loglik_dthp, loglik_seir : array-like
        Per-step log-evidence for each model (already logged; NOT raw evidence).
    window : int
        Sliding window length (in time steps).
    t_burning : int
        Number of initial steps accumulated without windowing.

    Returns
    -------
    w_dthp, w_seir : ndarray
        Posterior model weights over time (sum to 1 at every step).
    """
    loglik_dthp = np.asarray(loglik_dthp)
    loglik_seir = np.asarray(loglik_seir)
    T = len(loglik_dthp)

    logZ_dthp = np.zeros(T)
    logZ_seir = np.zeros(T)
    logZ_dthp[0] = loglik_dthp[0]
    logZ_seir[0] = loglik_seir[0]

    for t in range(1, T):
        if t < t_burning:
            logZ_dthp[t] = logZ_dthp[t - 1] + loglik_dthp[t]
            logZ_seir[t] = logZ_seir[t - 1] + loglik_seir[t]
        else:
            start = max(t_burning, t - window + 1)
            logZ_dthp[t] = np.sum(loglik_dthp[start:t + 1])
            logZ_seir[t] = np.sum(loglik_seir[start:t + 1])

    w_dthp = np.zeros(T)
    w_seir = np.zeros(T)
    for t in range(T):
        max_log = max(logZ_dthp[t], logZ_seir[t])
        wd = np.exp(logZ_dthp[t] - max_log)
        ws = np.exp(logZ_seir[t] - max_log)
        norm = wd + ws
        w_dthp[t] = wd / norm
        w_seir[t] = ws / norm

    return w_dthp, w_seir


def compute_model_average(matrix_dict_dthp, matrix_dict_seir, pi_dthp, pi_seir, random_state=None):
    """
    Build model-averaged predictive particle sets using deterministic
    largest-remainder allocation, given time-varying model weights.

    Parameters
    ----------
    matrix_dict_dthp, matrix_dict_seir : dict[str, ndarray]
        Per-variable N_particles x T arrays of predictive particles for
        each model (as produced by `src.visualization.trace_smc`).
    pi_dthp, pi_seir : array-like, shape (T,)
        Posterior model weights over time (e.g. from `compute_window_weights`).
    random_state : int or None
        Random seed for the resampling step.

    Returns
    -------
    averaged_dict : dict[str, ndarray]
        Model-averaged particle matrices (N_particles x T) for every
        variable common to both models.
    """
    rng = np.random.default_rng(random_state)
    pi_dthp = np.asarray(pi_dthp)
    pi_seir = np.asarray(pi_seir)
    averaged_dict = {}

    for key in matrix_dict_dthp.keys():
        if key not in matrix_dict_seir:
            continue

        particles_dthp = np.asarray(matrix_dict_dthp[key])
        particles_seir = np.asarray(matrix_dict_seir[key])
        Np, T = particles_dthp.shape

        if particles_seir.shape != (Np, T):
            raise ValueError(f"Particle matrices for '{key}' must have the same shape.")
        if len(pi_dthp) != T or len(pi_seir) != T:
            raise ValueError("Posterior model probabilities must have length T.")

        avg_particles = np.empty((Np, T), dtype=particles_dthp.dtype)

        for t in range(T):
            probabilities = np.array([pi_dthp[t], pi_seir[t]], dtype=float)
            probabilities /= probabilities.sum()

            # Largest-remainder allocation: floor first, then hand out the
            # leftover particles to whichever model has the biggest fractional part.
            expected_counts = Np * probabilities
            allocations = np.floor(expected_counts).astype(int)
            remaining = Np - allocations.sum()

            if remaining > 0:
                fractional_remainders = expected_counts - allocations
                order = np.argsort(-fractional_remainders)
                for k in order[:remaining]:
                    allocations[k] += 1

            n_dthp, n_seir = allocations
            idx_dthp = rng.choice(Np, size=n_dthp, replace=True)
            idx_seir = rng.choice(Np, size=n_seir, replace=True)

            avg_particles[:, t] = np.concatenate(
                [particles_dthp[idx_dthp, t], particles_seir[idx_seir, t]]
            )

        averaged_dict[key] = avg_particles

    return averaged_dict
