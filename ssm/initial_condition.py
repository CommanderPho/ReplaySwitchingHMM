import numpy as np

def make_uniform_initial(n_states, n_position_bins):
    initial_conditions = np.ones(
        (n_states, n_position_bins, 1), dtype=np.float64)
    return initial_conditions / np.sum(initial_conditions)