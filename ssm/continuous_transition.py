import numpy as np

def _normalize_row_probability(x: np.ndarray) -> np.ndarray:
    """Ensure the state transition matrix rows sum to 1.

    Parameters
    ----------
    x : np.ndarray, shape (n_rows, n_cols)

    Returns
    -------
    normalized_x : np.ndarray, shape (n_rows, n_cols)

    """
    x /= x.sum(axis=1, keepdims=True)
    x[np.isnan(x)] = 0
    return x

def make_uniform_transition(n_position_bins):
    transition_matrix = np.ones((n_position_bins, n_position_bins))
    return _normalize_row_probability(transition_matrix)

def make_identity_transition(n_position_bins):
    transition_matrix = np.identity(n_position_bins)
    return _normalize_row_probability(transition_matrix)

def make_driftdiffusion_transition(n_position_bins, lamb, sig, dt):  
    grid = np.arange(n_position_bins, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(grid, grid, indexing='ij')
    dist = grid_y - grid_x - lamb * dt
    # Exact Gaussian expression
    transition_matrix = (
        1 / (((2 * np.pi) ** 0.5) * (sig * (dt ** 0.5))) *
        np.exp(-np.square(dist) / (2 * np.square(sig) * dt)))
    
    # See below numerical considerations
    if (np.sum(transition_matrix, axis=1) > 1).any():
        rows = np.sum(transition_matrix, axis=1) > 1
        transition_matrix[rows] = \
            transition_matrix[rows] / np.sum(transition_matrix[rows], axis=1, keepdims=True)
    
    # Out of boundary cases
    ######
    # mu_tplus1 = grid_x + lamb * dt # mean of location
    # oob_rows = np.where((mu_tplus1[:, 0] < 0) | (mu_tplus1[:, 0] > n_position_bins-1))[0]
    #     # out of boundary row indicies
    ######
    # oob_rows = np.sum(transition_matrix, axis=1) < 0.4
    # transition_matrix[oob_rows, :] = 1/n_position_bins
    ######
    # oob_rows = np.max(transition_matrix, axis=1) < 0.01
    # transition_matrix[oob_rows, :] = 1/n_position_bins
    ######
    transition_matrix += 1e-10
    return transition_matrix
    # return _normalize_row_probability(transition_matrix)

    # Numerical considerations:
    # 1. (not sure?) do not normalize each row to 1
    #    to correctly calculate likelihood at boundaries.
    # 2. sigma cannot be too small.
    #    when sigma_dd is very small, the pdf above is incorrect numerically
    #    make sure square(sigma_dd)*time_window > 0.5 bin.
    
    