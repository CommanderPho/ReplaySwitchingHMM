import numpy as np
from scipy.special import xlogy

def poisson_log_likelihood(spikes, conditional_intensity):
    """Probability of parameters given spiking at a particular time.

    Parameters
    ----------
    spikes : np.ndarray, shape (n_time,)
        Indicator of spike or no spike at current time.
    conditional_intensity : np.ndarray, shape (n_place_bins,)
        Instantaneous probability of observing a spike

    Returns
    -------
    poisson_log_likelihood : array_like, shape (n_time, n_place_bins)

    """
    return (
        xlogy(spikes[:, np.newaxis], conditional_intensity[np.newaxis, :])
        - conditional_intensity[np.newaxis, :]
    )

def estimate_spiking_likelihood(spikes, conditional_intensity):
    """

    Parameters
    ----------
    spikes : np.ndarray, shape (n_time, n_neurons)
    conditional_intensity : np.ndarray, shape (n_bins, n_neurons)

    """
    n_time = spikes.shape[0]
    n_bins = conditional_intensity.shape[0]
    log_likelihood = np.zeros((n_time, n_bins))

    conditional_intensity = np.clip(conditional_intensity, a_min=1e-15, a_max=None)

    for is_spike, ci in zip(spikes.T, conditional_intensity.T):
        log_likelihood += poisson_log_likelihood(is_spike, ci)

    return log_likelihood