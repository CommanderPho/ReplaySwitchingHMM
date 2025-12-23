import numpy as np

def make_diagonal_transition(n_states, diagonal_value):  
    state_transition = np.identity(n_states) * diagonal_value
    is_off_diag = ~np.identity(n_states, dtype=bool)
    state_transition[is_off_diag] = (1 - diagonal_value) / (n_states - 1)
    return state_transition