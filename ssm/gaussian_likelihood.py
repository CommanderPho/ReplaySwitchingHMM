import numpy as np

def estimate_dv_likelihood(observations, emission_matrix):
    """
    Calculate the log-likelihood for decision variable observations.

    Parameters
    ----------
    observations : np.ndarray, shape (n_time, n_observation_bins)
        One-hot encoded decision variable observations.
    emission_matrix : np.ndarray, shape (n_latent_bins, n_observation_bins)
        Emission probability matrix, where rows correspond to latent bins
        and columns correspond to observation bins.

    Returns
    -------
    log_likelihood : np.ndarray, shape (n_time, n_latent_bins)

    """
    n_time = observations.shape[0]
    n_latent_bins = emission_matrix.shape[0]

    emission_matrix = np.clip(emission_matrix, a_min=1e-15, a_max=None)
        # clip the emission matrix to avoid log(0) issues

    log_likelihood = np.zeros((n_time, n_latent_bins))
    weighted_emission_probs = observations @ emission_matrix.T
        # extract the probabilities for the observed bin across all latent bins
    log_likelihood = np.log(weighted_emission_probs)    
    log_likelihood[np.isneginf(log_likelihood)] = 0
        # handle cases where observations are all zeros (log(0) issues)

    return log_likelihood
