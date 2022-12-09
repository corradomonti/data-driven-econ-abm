from tfutils import softmax_by_index
from utils import homophily_matrix

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from collections import namedtuple
import logging

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

ε = 1E-10

PHI_MEAN = 30. * 12 * 1.3 * 0.4 # Years, months, income earners per household, rent over income
PHI_STD = PHI_MEAN * np.sqrt(
    (10 / 30) ** 2 + 
    (0 / 12) ** 2 + 
    (0.3 / 1.3) ** 2 +
    (0.1 / 0.4) ** 2
)

def build_model(L, K, N, initial_Gammak,
        price_error_stddev=1., num_deals_error_stddev=1., 
                # Probability to put a house on sale
                alpha = 0.1,
                # Weight of attractiveness
                beta = 0.9,
                # Adjustment rate of reservation prices
                delta = 0.1,
                # Bargaining power of sellers
                nu = 0.1,
                # Exponent
                tau = 2.,
                # Use mean field approximation or leave placeholders?
                use_Nb_mean=True,
                # Number of samples to take from all the possible Dbs (MUST BE SET)
                num_Db_values = None,
                # Use homophily instead of wealth to determine attractiveness. 
                homophily=None,
                # This needs to be either None (use wealth and not homophily) or an integer κ.
                # κ is the maximum distance around class i where they are considered homophiliac.
                learnable_phi=False,
                # Allow to use M0 instead of its logits for debug purposes
                use_M0_without_logits=False,
                # Weight of φ prior in loss. Can be float, or 'auto' -> mean of all P & Nd together.
                phi_weight=1.,
                # Use relative error instead of absolut error for Nd and P.
                use_relative_error=False,
        ):
    logging.info("Building TfModel...")
    assert num_Db_values is not None
    assert isinstance(phi_weight, (int, float)) or phi_weight == 'auto'
    tf.reset_default_graph()

    # Constant and params
    
    Y = tf.placeholder(tf.float32, shape=(K, ), name="Y")
    A0 = tf.placeholder(tf.float32, shape=(L, ), name="A0")
    Gammak = tf.placeholder(tf.float32, shape=(K, ), name="Gammak")
    Q = tf.placeholder(tf.float32, name="Q") # Num incomers

    # Observable

    R_prev = tf.placeholder(tf.float32, shape=(L, ), name="R_prev")
    given_Nd = tf.placeholder(tf.float32, shape=(L, ), name="Nd")
    P_prev = tf.placeholder(tf.float32, shape=(L, ), name="P_prev")

    M_delta_prev = tf.placeholder(tf.float32, shape=(L, K), name="M_delta")
    given_P = tf.placeholder(tf.float32, shape=(L, ), name='P_input')
    
    
    phi_min = tf.placeholder(tf.float32, shape=(), name='phi_min')
    
    # Variable coefficient between price and income
    if learnable_phi:
        phi_unbounded = tf.get_variable('phi', shape=(),
            initializer=tf.constant_initializer(PHI_MEAN), dtype=tf.float32)
        phi = tf.maximum(phi_min, phi_unbounded)
        # This should use tfp.distributions.Masked to mask out values below phi_min
        phi_prior = tfp.distributions.Normal(
            tf.constant(PHI_MEAN, dtype=tf.float32), tf.constant(PHI_STD, dtype=tf.float32))
        if phi_weight == 'auto':
            phi_weight = tf.reduce_mean(tf.concat([given_P, given_Nd], 0))
        phi_logp = phi_weight * phi_prior.log_prob(phi)
        Y_tilde = phi * Y
    else:
        phi_unbounded = phi = tf.get_variable('phi', shape=(), initializer=tf.constant_initializer(1.), dtype=tf.float32)
        phi_logp = tf.constant(0.)
        Y_tilde = Y
    
    # Variable M0
    if use_M0_without_logits:
        M0 = M0_dirichlet = M0_logit = tf.get_variable('M0_logit', shape=(L, K))
    else:
        M0_logit = tf.get_variable('M0_logit', shape=(L, K),
                               initializer=tf.constant_initializer(
                                   np.tile(np.log(initial_Gammak), [L, 1])
                               ))
        M0_dirichlet = tf.nn.softmax(M0_logit, axis=1, name='M0_dirichlet')
        M0 = tf.maximum(0., N * M0_dirichlet, name='M0')
    
    M_prev = tf.maximum(0., tf.add(M0, M_delta_prev), name='M_t-1')
    
    # Number of sellers (deterministic from input)
    population_distr = M_prev / (tf.reduce_sum(M_prev, 1)[:, np.newaxis] + ε)
    Ds = tf.multiply(tf.tile(given_Nd[:, np.newaxis], [1, K]), population_distr, name='Ds')
    
    # Attractiveness (deterministic)
    # This part needs to be done in log scale to avoid numerical errors.
    # See notebook/dev-log-version-of-Pi-from-A.ipynb and
    # notebook/dev-homophily-attractiveness.ipynb to understand why the following works.
    if homophily is None:
        normalized_M_prev = M_prev / tf.reduce_sum(M_prev, 1)[:, tf.newaxis]
        log_AY_unnorm = tau * tf.log(tf.einsum('ik,k->i', normalized_M_prev, Y_tilde) + ε)
        log_AY = np.log(L) + tf.nn.log_softmax(log_AY_unnorm)
        log_A_vector = tf.log(A0) + log_AY # This is a (L, )-shaped vector
        log_A = tf.tile(log_A_vector[:, np.newaxis], [1, K]) # This is a (L, K)-shaped matrix
    else:
        H_matrix = tf.constant(homophily_matrix(K, homophily))
        log_AH_unnorm = tau * tf.log(tf.einsum('xk,kj->xj', 1 + M_prev, H_matrix) + 1)
        log_AH = np.log(L) + tf.nn.log_softmax(log_AH_unnorm, 0)
        log_A = tf.log(A0[:, np.newaxis]) + log_AH # This is a (L, K)-shaped matrix
    affordability = tf.tile(Y_tilde[np.newaxis, :], [L, 1]) - tf.tile(P_prev[:, np.newaxis], [1, K])
    log_Pi_unnorm = beta * log_A + (1 - beta) * tf.log(tf.maximum(0., affordability) + ε)
    Pi_unnorm = tf.where(affordability > 0., tf.nn.softmax(log_Pi_unnorm, axis=0), np.zeros((L, K)))
    Pi = Pi_unnorm / (tf.reduce_sum(Pi_unnorm, 0) + ε)
    
    # Nb Expected value
    Nb_mean = Q * Gammak * Pi
    Nb_mean.set_shape((L, K))
    Nb = Nb_mean if use_Nb_mean else tf.placeholder(tf.float32, shape=(L, K), name="Nb")
    
    # Deterministic
    Ns = R_prev + (N-R_prev) * alpha
    Nd = tf.minimum(tf.reduce_sum(Nb, axis=1), Ns)
    Nd_error_prior = tfp.distributions.Normal(0, num_deals_error_stddev)
    Nd_error = (Nd - given_Nd) / given_Nd if use_relative_error else (Nd - given_Nd)
    Nd_prior_logp = Nd_error_prior.log_prob(Nd_error)

    BS_balance = tf.reduce_sum(Nb, axis=1) / Ns

    Ps = P_prev * (1-delta+delta*tf.math.tanh(BS_balance))

    income_left = tf.tile(Y_tilde[np.newaxis, :], [L, 1]) - tf.tile(Ps[:, np.newaxis], [1, K])
    prob_by_class_unnorm = tf.sqrt(tf.maximum(income_left, 0)) * Nb
    prob_by_class = tf.divide(prob_by_class_unnorm + ε,
        tf.reduce_sum(prob_by_class_unnorm, 1)[:, np.newaxis] + ε, name='prob_by_class')
        
    # Prior
    P_error_prior = tfp.distributions.Normal(0, price_error_stddev)
    
    Db_distributions = tfp.distributions.Multinomial(total_count=given_Nd, probs=prob_by_class)
    
    Db_locs = tf.placeholder(tf.int32, shape=(num_Db_values, ), name="Db_locs")
    Db_values = tf.placeholder(tf.float32, shape=(num_Db_values, K), name="Db_values")
    Db_weights = tf.placeholder(tf.float32, shape=(num_Db_values, ), name="Db_weights")
    
    # Operations from Db to Prices, they are all independent between rows of Db (i.e., locations)
    Db_norm = (
        tf.transpose(Db_values) / (tf.transpose(tf.maximum(tf.reduce_sum(Db_values, 1), 1)) + ε))
    Pb_per_loc = tf.einsum('k,kx->x', Y_tilde, Db_norm, name='Pb')
    Ps_per_loc = tf.gather(Ps, Db_locs)
    P_per_loc = tf.add(nu * Pb_per_loc, (1-nu) * Ps_per_loc, name='P')
    
    # Compute loss from P and likelihood from Db for each Db_value.
    Db_log_likelihoods = []
    all_P = []
    P_errors = []
    for sample_loc, sample_P, sample_Db in zip(
                                tf.unstack(Db_locs), tf.unstack(P_per_loc), tf.unstack(Db_values)):
        # Compute error on P for this sample (M step).
        P_error = ((sample_P - given_P[sample_loc]) / given_P[sample_loc]
            ) if use_relative_error else (sample_P - given_P[sample_loc])
        P_errors.append(P_error)
        # Compute likelihood of this sample (E step).
        sample_Db_log_likelihood = tf.reduce_sum(Db_distributions.log_prob(sample_Db)[sample_loc])
        Db_log_likelihood_considering_Nb = tf.cond(    # Db_{l, k} must be <= Nb{l, k} for each l, k
            tf.math.reduce_all(tf.less_equal(sample_Db, Nb[sample_loc]+1)), # Since <= is element-wise
            lambda: sample_Db_log_likelihood, lambda: -10E6 # pylint: disable=W0640
        )
        Db_log_likelihoods.append(Db_log_likelihood_considering_Nb)
        # Aggregate prices for each sample (debug).
        all_P.append(sample_P)
    
    # Aggregate loss function over the samples (M step).
    P_prior_logps = P_error_prior.log_prob(tf.stack(P_errors))
    P_prior_logp = tf.reduce_sum(Db_weights * P_prior_logps, name='loss')
    assert P_prior_logp.shape == phi_logp.shape == ()
    assert Nd_prior_logp.shape == (L, )
    loss = -(P_prior_logp + tf.reduce_sum(Nd_prior_logp) + phi_logp)
    
    # Compute the likelihood of each sample (E step).
    Db_log_likelihood_unnorm = tf.stack(Db_log_likelihoods)
    Db_log_likelihood = softmax_by_index(Db_log_likelihood_unnorm, Db_locs, L)
    Db_likelihood = tf.exp(Db_log_likelihood)
    
    # Aggregate debug values of each sample.
    P = tf.stack(all_P)
    
    logging.info("TfModel was built.")
    
    # Construct TfModel structure.
    ins = namedtuple("ins",
        "Db_values, Db_weights, Db_locs, Y, A0, Q, Gammak, given_Nd, R_prev, P_prev, M_delta_prev, given_P, phi_min")\
        (Db_values, Db_weights, Db_locs, Y, A0, Q, Gammak, given_Nd, R_prev, P_prev, M_delta_prev, given_P, phi_min)
        
    outs = namedtuple("outs",
        "Ns, Nb, Nd, P, M0, M0_logit, Ds, phi, phi_unbounded")\
        (Ns, Nb, Nd, P, M0, M0_logit, Ds, phi, phi_unbounded)
        
    debug = namedtuple("debug",
        "prob_by_class, P_prior_logp, Pi, log_A, affordability, log_Pi_unnorm, Nd_prior_logp, Db_distributions, P_per_loc, log_AY_unnorm, M0_dirichlet, M_prev, Y_tilde, P_prior_logps")\
        (prob_by_class, P_prior_logp, Pi, log_A, affordability, log_Pi_unnorm, Nd_prior_logp, Db_distributions, P_per_loc, log_AY_unnorm, M0_dirichlet, M_prev, Y_tilde, P_prior_logps)
         
    return namedtuple("Model",
        "ins, outs, loss, Db_likelihood, Db_log_likelihood, debug, learnable_phi")\
        (ins, outs, loss, Db_likelihood, Db_log_likelihood, debug, learnable_phi)
    
