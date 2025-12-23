import numpy as np
from typing import Tuple

def normalize_to_probability(distribution: np.ndarray) -> np.ndarray:
    """Ensure the distribution integrates to 1 so that it is a probability
    distribution.

    Parameters
    ----------
    distribution : np.ndarray

    Returns
    -------
    normalized_distribution : np.ndarray

    """
    return distribution / np.nansum(distribution)

def scaled_likelihood(log_likelihood: np.ndarray, axis: int = 1) -> np.ndarray:
    """Scale the likelihood so the maximum value is 1.

    Parameters
    ----------
    log_likelihood : np.ndarray, shape (n_time, n_bins)
    axis : int

    Returns
    -------
    scaled_log_likelihood : np.ndarray, shape (n_time, n_bins)

    """
    max_log_likelihood = np.nanmax(log_likelihood, axis=axis, keepdims=True)
    # If maximum is infinity, set to zero
    if max_log_likelihood.ndim > 0:
        max_log_likelihood[~np.isfinite(max_log_likelihood)] = 0.0
    elif not np.isfinite(max_log_likelihood):
        max_log_likelihood = 0.0

    # Maximum likelihood is always 1
    likelihood = np.exp(log_likelihood - max_log_likelihood)
    # avoid zero likelihood
    likelihood += np.spacing(1, dtype=likelihood.dtype)
    return likelihood

def compute_causal(
    initial_conditions: np.ndarray,
    continuous_state_transition: np.ndarray,
    discrete_state_transition: np.ndarray,
    likelihood: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Adaptive filter to iteratively calculate the posterior probability
    of a state variable using past information.

    Parameters
    ----------
    initial_conditions : np.ndarray, shape (n_states, n_bins, 1)
    continuous_state_transition : np.ndarray, shape (n_states, n_states, n_bins, n_bins)
    discrete_state_transition : np.ndarray, shape (n_states, n_states)
    likelihood : np.ndarray, shape (n_time, n_states, n_bins, 1)

    Returns
    -------
    causal_posterior : np.ndarray, shape (n_time, n_states, n_bins, 1)
    log_data_likelihood : float

    """
    n_time, n_states, n_bins, _ = likelihood.shape
    posterior = np.zeros_like(likelihood)

    posterior[0] = initial_conditions.copy() * likelihood[0]
    norm = np.nansum(posterior[0])
    log_data_likelihood = np.log(norm)
    posterior[0] /= norm

    for k in np.arange(1, n_time):
        prior = np.zeros((n_states, n_bins, 1))
        for state_k in np.arange(n_states):
            for state_k_1 in np.arange(n_states):
                prior[state_k, :] += (
                    discrete_state_transition[state_k_1, state_k]
                    * continuous_state_transition[state_k_1, state_k].T
                    @ posterior[k - 1, state_k_1]
                )
        posterior[k] = prior * likelihood[k]
        norm = np.nansum(posterior[k])
        log_data_likelihood += np.log(norm)
        posterior[k] /= norm

    return posterior, log_data_likelihood

def compute_acausal(
    causal_posterior: np.ndarray,
    continuous_state_transition: np.ndarray,
    discrete_state_transition: np.ndarray,
) -> np.ndarray:
    """Uses past and future information to estimate the state.

    Parameters
    ----------
    causal_posterior : np.ndarray, shape (n_time, n_states, n_bins, 1)
    continuous_state_transition : np.ndarray, shape (n_states, n_states, n_bins, n_bins)
    discrete_state_transition : np.ndarray, shape (n_states, n_states)

    Return
    ------
    acausal_posterior : np.ndarray, shape (n_time, n_states, n_bins, 1)

    """
    acausal_posterior = np.zeros_like(causal_posterior)
    acausal_posterior[-1] = causal_posterior[-1].copy()
    n_time, n_states, n_bins, _ = causal_posterior.shape
    eps = np.spacing(1)

    for k in np.arange(n_time - 2, -1, -1):
        # Prediction Step -- p(x_{k+1}, I_{k+1} | y_{1:k})
        prior = np.zeros((n_states, n_bins, 1))
        for state_k_1 in np.arange(n_states):
            for state_k in np.arange(n_states):
                prior[state_k_1, :] += (
                    discrete_state_transition[state_k, state_k_1]
                    * continuous_state_transition[state_k, state_k_1].T
                    @ causal_posterior[k, state_k]
                )

        # Backwards Update
        weights = np.zeros((n_states, n_bins, 1))
        ratio = np.exp(np.log(acausal_posterior[k + 1] + eps) - np.log(prior + eps))
        for state_k in np.arange(n_states):
            for state_k_1 in np.arange(n_states):
                weights[state_k] += (
                    discrete_state_transition[state_k, state_k_1]
                    * continuous_state_transition[state_k, state_k_1]
                    @ ratio[state_k_1]
                )

        acausal_posterior[k] = normalize_to_probability(weights * causal_posterior[k])

    return acausal_posterior