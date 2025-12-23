import numpy as np
from ssm.initial_condition import make_uniform_initial
from ssm.continuous_transition import (
    make_uniform_transition,
    make_identity_transition,
    make_driftdiffusion_transition)
from ssm.discrete_transition import make_diagonal_transition
from ssm.spiking_likelihood import estimate_spiking_likelihood
from ssm.gaussian_likelihood import estimate_dv_likelihood
from ssm.bayesian import (
    scaled_likelihood,
    compute_causal,
    compute_acausal)


class SSM():
    def __init__(
        self,
        place_fields, dt,
        gamma,
        initial_condition_type,
        discrete_transition_type,
        continuous_transition_types,
        dtype = np.float64):
        """

        Parameters
        ----------
        place_fields : np.ndarray, shape (n_pos_bins, n_neurons)
        init_condition : str
        """
        self.dtype = dtype
        self.place_fields = place_fields
        self.dt = dt
        self.gamma = gamma
        self.conditional_intensity = gamma * place_fields * dt # rate param of Poisson distribution
        self.n_pos_bins = place_fields.shape[0]
        self.n_neurons = place_fields.shape[1]
        self.n_states = continuous_transition_types.shape[0]

        self.initial_conditions = self.calc_initial_conditions(initial_condition_type)
        self.discrete_state_transition = self.calc_discrete_state_transition(discrete_transition_type)
        self.continuous_state_transition = self.calc_continuous_state_transition(continuous_transition_types)

    def calc_initial_conditions(self, ic_type):
        """Constructs the initial probability for the state and each spatial bin."""

        if ic_type['name'] == 'uniform':
            initial_conditions = make_uniform_initial(self.n_states, self.n_pos_bins)
        else:
            raise NotImplementedError
        return initial_conditions

    def calc_continuous_state_transition(self, ct_types):
        """Constructs the transition matrices for the continuous states."""

        assert ct_types.shape[0] == self.n_states
        assert ct_types.shape[1] == self.n_states
        continuous_transition_matrix = \
            np.full((self.n_states, self.n_states, self.n_pos_bins, self.n_pos_bins), np.nan, dtype=self.dtype)

        for i in range(self.n_states):
            for j in range(self.n_states):
                ct_type = ct_types[i, j]
                if ct_type['name'] == 'uniform':
                    continuous_transition_matrix[i, j, :, :] = \
                        make_uniform_transition(self.n_pos_bins)
                elif ct_type['name'] == 'identity':
                    continuous_transition_matrix[i, j, :, :] = \
                        make_identity_transition(self.n_pos_bins)
                elif ct_type['name'] == 'driftdiffusion':
                    continuous_transition_matrix[i, j, :, :] = \
                        make_driftdiffusion_transition(self.n_pos_bins, ct_type['lamb'], ct_type['sig'], self.dt)
                else:
                    raise NotImplementedError
        return continuous_transition_matrix
    
    def calc_discrete_state_transition(self, dt_type):
        """Constructs the transition matrix for the discrete states."""

        if dt_type['name'] == 'diagonal':
            discrete_transition_matrix = make_diagonal_transition(self.n_states, dt_type['diagonal_value'])
        else:
            raise NotImplementedError
        return discrete_transition_matrix

    def fit_spiking_likelihood(self, spike_raster):
        # log_likelihood = estimate_spiking_likelihood(
        #     spike_raster, self.place_fields)
        log_likelihood = estimate_spiking_likelihood(
            spike_raster, self.conditional_intensity)
        return log_likelihood
    
    def fit_causal_posterior(self, log_likelihood):
        causal_posterior, data_log_likelihood = compute_causal(
            initial_conditions=self.initial_conditions.astype(self.dtype),
            continuous_state_transition=self.continuous_state_transition.astype(self.dtype),
            discrete_state_transition=self.discrete_state_transition.astype(self.dtype),
            likelihood=log_likelihood.astype(self.dtype))
        return causal_posterior, data_log_likelihood

    def fit_acausal_posterior(self, causal_posterior):
        acausal_posterior = compute_acausal(
            causal_posterior=causal_posterior.astype(self.dtype),
            continuous_state_transition=self.continuous_state_transition.astype(self.dtype),
            discrete_state_transition=self.discrete_state_transition.astype(self.dtype))
        return acausal_posterior

    def fit_inference(self, spike_raster):
        """

        Parameters
        ----------
        spike_raster : np.ndarray, shape (n_time, n_neurons)
        """
        # 1. Emission likelihood
        log_likelihood = self.fit_spiking_likelihood(spike_raster)
        # 2. Repeat for all discrete states
        n_time = spike_raster.shape[0]
        log_likelihood_allstates = np.full(
            (n_time, self.n_states, self.n_pos_bins, 1), np.nan, dtype=self.dtype)
        for state_ind in range(self.n_states):
            log_likelihood_allstates[:, state_ind, :, :] = \
                log_likelihood[:, :, np.newaxis]
        # zhongxuan: removed scaled_likelihood here
        # likelihood_allstates = scaled_likelihood(log_likelihood_allstates, axis=(1, 2))
        likelihood_allstates = np.exp(log_likelihood_allstates)
        likelihood_allstates += np.spacing(1, dtype=likelihood_allstates.dtype)
            # this returns likelihood, not log-likelihood
        likelihood_allstates[np.isnan(likelihood_allstates)] = 0.0
        # 3. Causal filtering
        causal_posterior, data_log_likelihood = self.fit_causal_posterior(likelihood_allstates)
        # 4. Acausal smoothing
        acausal_posterior = self.fit_acausal_posterior(causal_posterior)
        return causal_posterior, acausal_posterior, data_log_likelihood

class SSM_DV(SSM):
    def __init__(
        self,
        place_fields, dt,
        gamma,
        initial_condition_type,
        discrete_transition_type,
        continuous_transition_types,
        dtype=np.float64
    ):
        super().__init__(place_fields, dt, gamma, initial_condition_type, discrete_transition_type, continuous_transition_types, dtype)
        self.conditional_intensity = place_fields

    def fit_dv_likelihood(self, spike_raster):
        # log_likelihood = estimate_spiking_likelihood(
        #     spike_raster, self.place_fields)
        log_likelihood = estimate_dv_likelihood(
            spike_raster, self.conditional_intensity)
        return log_likelihood

    def fit_inference(self, spike_raster):
        """

        Parameters
        ----------
        spike_raster : np.ndarray, shape (n_time, n_neurons)
        """
        # 1. Emission likelihood
        log_likelihood = self.fit_dv_likelihood(spike_raster)
        # 2. Repeat for all discrete states
        n_time = spike_raster.shape[0]
        log_likelihood_allstates = np.full(
            (n_time, self.n_states, self.n_pos_bins, 1), np.nan, dtype=self.dtype)
        for state_ind in range(self.n_states):
            log_likelihood_allstates[:, state_ind, :, :] = \
                log_likelihood[:, :, np.newaxis]
        # zhongxuan: removed scaled_likelihood here
        # likelihood_allstates = scaled_likelihood(log_likelihood_allstates, axis=(1, 2))
        likelihood_allstates = np.exp(log_likelihood_allstates)
        likelihood_allstates += np.spacing(1, dtype=likelihood_allstates.dtype)
            # this returns likelihood, not log-likelihood
        likelihood_allstates[np.isnan(likelihood_allstates)] = 0.0
        # 3. Causal filtering
        causal_posterior, data_log_likelihood = self.fit_causal_posterior(likelihood_allstates)
        # 4. Acausal smoothing
        acausal_posterior = self.fit_acausal_posterior(causal_posterior)
        return causal_posterior, acausal_posterior, data_log_likelihood