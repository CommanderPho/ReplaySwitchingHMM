# Initial condition type presets
def ic_uniform(**kwargs):
    d = {'name': 'uniform'}
    d.update(kwargs)
    return d

# Discrete transition type presets
def dt_diagonal(**kwargs):
    d = {'name':'diagonal', 'diagonal_value':None}
    d.update(kwargs)
    return d

# Continuous transition type presets
def ct_fragmented(**kwargs):
    d = {'name':'uniform'}
    d.update(kwargs)
    return d

def ct_stationary(**kwargs):
    d = {'name':'identity'}
    d.update(kwargs)
    return d

def ct_driftdiffusion(**kwargs):
    d = {'name':'driftdiffusion', 'lamb':None, 'sig':None}
    d.update(kwargs)
    return d