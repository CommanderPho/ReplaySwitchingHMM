import numpy as np
from ssm.ssm import SSM
from ssm.utils import (
    ic_uniform,
    dt_diagonal,
    ct_fragmented, ct_stationary, ct_driftdiffusion)

def perform_inference(params, place_fields, spike_raster, dt, model_type):
    diagonal_value, lamb1, lamb2, sig = params
    # diagonal_value = 0.99 # 0.98
    gamma = 1.0 # 2.5
    if model_type == 'hybrid_drift_diffusion':
        ic_type = ic_uniform()
        dt_type = dt_diagonal(diagonal_value=diagonal_value)
        ######
        # ct_row = np.array(
        #     [ct_driftdiffusion(lamb=lamb1, sig=sig),
        #      ct_driftdiffusion(lamb=-1 * lamb2, sig=sig),
        #      ct_fragmented(),
        #      ct_stationary()], dtype=object)
        ######
        ct_row = np.array(
            [ct_driftdiffusion(lamb=lamb1, sig=sig),
             ct_fragmented(),
             ct_stationary()], dtype=object)
        ######
        ct_types = np.tile(ct_row, (ct_row.shape[0], 1))
    elif model_type == 'hybrid_diffusion':
        ic_type = ic_uniform()
        dt_type = dt_diagonal(diagonal_value=diagonal_value)
        ct_row = np.array(
            [ct_driftdiffusion(lamb=0, sig=sig),
             ct_driftdiffusion(lamb=0, sig=sig),
             ct_fragmented(),
             ct_stationary()], dtype=object)
        ct_types = np.tile(ct_row, (ct_row.shape[0], 1))
    elif model_type == 'drift_diffusion':
        ic_type = ic_uniform()
        dt_type = dt_diagonal(diagonal_value=diagonal_value)
        ct_row = np.array(
            [ct_driftdiffusion(lamb=lamb1, sig=sig),
             ct_driftdiffusion(lamb=lamb1, sig=sig)], dtype=object)
        ct_types = np.tile(ct_row, (ct_row.shape[0], 1))
    elif model_type == 'diffusion':
        ic_type = ic_uniform()
        dt_type = dt_diagonal(diagonal_value=diagonal_value)
        ct_row = np.array(
            [ct_driftdiffusion(lamb=0, sig=sig),
             ct_driftdiffusion(lamb=0, sig=sig)], dtype=object)
        ct_types = np.tile(ct_row, (ct_row.shape[0], 1))
    elif model_type == 'frag_stat':
        ic_type = ic_uniform()
        dt_type = dt_diagonal(diagonal_value=diagonal_value)
        ct_row = np.array(
            [ct_fragmented(),
             ct_stationary()], dtype=object)
        ct_types = np.tile(ct_row, (ct_row.shape[0], 1))
    elif model_type == 'frag':
        ic_type = ic_uniform()
        dt_type = dt_diagonal(diagonal_value=diagonal_value)
        ct_row = np.array(
            [ct_fragmented(),
             ct_fragmented()], dtype=object)
        ct_types = np.tile(ct_row, (ct_row.shape[0], 1))
    elif model_type == 'stat':
        ic_type = ic_uniform()
        dt_type = dt_diagonal(diagonal_value=diagonal_value)
        ct_row = np.array(
            [ct_stationary(),
             ct_stationary()], dtype=object)
        ct_types = np.tile(ct_row, (ct_row.shape[0], 1))
    else:
        raise NotImplementedError
    model = SSM(place_fields, dt, gamma, ic_type, dt_type, ct_types)
    causal_posterior, acausal_posterior, data_log_likelihood = \
        model.fit_inference(spike_raster)
    return model, causal_posterior, acausal_posterior, data_log_likelihood

def neg_loglik(params, place_fields, spike_raster, dt, model_type):
    return -perform_inference(params, place_fields, spike_raster, dt, model_type)[3]