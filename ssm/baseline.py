import numpy as np
from scipy import special
from sklearn.linear_model import LinearRegression

# see decoding.py from RNN_Navigation_1
def bayesian_decoding(n, f_x, dx, dt, pos_bin_centers, epsilon=1e-10):
    """
    Input Args:
    n : spike raster, (neurons x timesteps)
    f_x : place fields, (neurons x position bins)
    dx : position bin size, scalar
    dt : time bin size, scalar
    pos_bin_centers : position bin centers for place fields, (position bins)

    Outputs:
    P_x_given_n : posterior of decoded positions, (timesteps x position bins)
    x_hat_Bayes : decoded positions, (timesteps)

    Note:
    No spatial prior for decoding (uniform prior).
    """
    f_x += epsilon # for numerical stability
    log_P_x_given_n = np.zeros((n.shape[1], f_x.shape[1]))

    for time_id in range(n.shape[1]):        
        log_P_x_given_n[time_id, :] = \
            np.sum((n[:, time_id:time_id+1]*np.log(dt*f_x)), axis=0) - \
            np.sum((np.log(special.factorial(n[:, time_id:time_id+1]))), axis=0) - \
            dt*np.sum(f_x, axis=0)
    
    x_hat_Bayes = pos_bin_centers[np.argmax(log_P_x_given_n, axis=1)]
    P_x_given_n = np.exp(log_P_x_given_n)
    P_x_given_n = P_x_given_n/np.sum(P_x_given_n, axis=1, keepdims=True)
    return P_x_given_n, x_hat_Bayes

def bayesian_decoding_smooth(position_bin_size, tau, n, f_x, epsilon, sigma_continuity):
    # no spatial prior
    x_t = np.arange(0, f_x.shape[1])*position_bin_size
    f_x += epsilon                                          # for numerical stability
    log_P_x_given_n = np.zeros((n.shape[1], f_x.shape[1]))  # time, position

    for time_id in range(n.shape[1]):
        if (time_id == 0) or (sigma_continuity is None):
            continuity_constraint = 0
        else:
            continuity_constraint = -np.square(x_t - x_hat_Bayes_tminus1)/(2*np.square(sigma_continuity))
        
        log_P_x_given_n[time_id, :] = \
            np.sum((n[:, time_id:time_id+1]*np.log(tau*f_x)), axis=0) - \
            np.sum((np.log(special.factorial(n[:, time_id:time_id+1]))), axis=0) - \
            tau*np.sum(f_x, axis=0) + \
            continuity_constraint
        
        x_hat_Bayes_tminus1 = np.argmax(log_P_x_given_n[time_id, :])*position_bin_size

    x_hat_Bayes = np.argmax(log_P_x_given_n, axis=1)*position_bin_size

    P_x_given_n = np.exp(log_P_x_given_n)
    P_x_given_n = P_x_given_n/np.sum(P_x_given_n, axis=1, keepdims=True)

    return P_x_given_n, x_hat_Bayes

def bayesian_decoding_acausal(position_bin_size, tau, n, f_x, epsilon, sigma_continuity):
    # no spatial prior
    x_t = np.arange(0, f_x.shape[1])*position_bin_size
    f_x += epsilon                                          # for numerical stability

    # Forward pass
    log_P_x_given_n_forward = np.zeros((n.shape[1], f_x.shape[1]))  # time, position
    for time_id in range(n.shape[1]):
        if (time_id == 0) or (sigma_continuity is None):
            continuity_constraint = 0
        else:
            continuity_constraint = -np.square(x_t - x_hat_Bayes_tminus1)/(2*np.square(sigma_continuity))
        
        log_P_x_given_n_forward[time_id, :] = \
            np.sum((n[:, time_id:time_id+1]*np.log(tau*f_x)), axis=0) - \
            np.sum((np.log(special.factorial(n[:, time_id:time_id+1]))), axis=0) - \
            tau*np.sum(f_x, axis=0) + \
            continuity_constraint
        
        x_hat_Bayes_tminus1 = np.argmax(log_P_x_given_n_forward[time_id, :])*position_bin_size
    
    # Backward pass
    log_P_x_given_n_backward = np.zeros((n.shape[1], f_x.shape[1]))
    for time_id in reversed(range(n.shape[1])):
        if time_id == n.shape[1] - 1 or sigma_continuity is None:
            continuity_constraint = 0
        else:
            continuity_constraint = -np.square(x_t - x_hat_Bayes_tplus1) / (2 * np.square(sigma_continuity))

        log_P_x_given_n_backward[time_id, :] = (
            np.sum(n[:, time_id:time_id + 1] * np.log(tau * f_x), axis=0) -
            np.sum(np.log(special.factorial(n[:, time_id:time_id + 1])), axis=0) -
            tau * np.sum(f_x, axis=0) +
            continuity_constraint
        )

        x_hat_Bayes_tplus1 = np.argmax(log_P_x_given_n_backward[time_id, :]) * position_bin_size

    # Combine
    log_P_x_given_n_smooth = (log_P_x_given_n_forward + log_P_x_given_n_backward) / 2
    x_hat_Bayes = np.argmax(log_P_x_given_n_smooth, axis=1) * position_bin_size

    P_x_given_n = np.exp(log_P_x_given_n_smooth)
    P_x_given_n = P_x_given_n / np.sum(P_x_given_n, axis=1, keepdims=True)

    return P_x_given_n, x_hat_Bayes

def simple_linear_regression(posterior, position_bin_size, tau):
    n_timesteps = posterior.shape[0]
    n_grid = posterior.shape[1]

    decoded_path = np.argmax(posterior, axis=1)
    reg = LinearRegression().fit(np.arange(n_timesteps).reshape(-1, 1), decoded_path)
    velocity = reg.coef_ * position_bin_size / tau

    return velocity

def weighted_correlation(posterior, position_bin_size, tau):
    """
    The time-position correlation coeffcient in posterior.
    """
    n_timesteps = posterior.shape[0]
    n_grid = posterior.shape[1]

    time_bins = (np.arange(n_timesteps)*tau)[:, None]
    position_bins = (np.arange(n_grid)*position_bin_size)[None, :]
    
    sum_posterior = np.sum(posterior)
    mean_time = np.sum(posterior*time_bins)/sum_posterior
    mean_position = np.sum(posterior*position_bins)/sum_posterior
    cov_time_time = np.sum(posterior*np.square(time_bins-mean_time))/sum_posterior
    cov_position_position = np.sum(posterior*np.square(position_bins-mean_position))/sum_posterior
    cov_time_position = np.sum(posterior*(time_bins-mean_time)*(position_bins-mean_position))/sum_posterior
    corr_time_position = cov_time_position/(1e-16+np.sqrt(cov_time_time*cov_position_position))

    return corr_time_position

def max_jump(posterior, position_bin_size):
    n_timesteps = posterior.shape[0]
    n_grid = posterior.shape[1]
    map_estimate = np.argmax(posterior, axis=1)
    jump = position_bin_size * \
        np.max(np.abs(map_estimate[1:] - map_estimate[:-1]))
    return jump