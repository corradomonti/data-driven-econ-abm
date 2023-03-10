from sample import get_Db_per_loc
import tfmodel
from utils import ConvergenceChecker

import numpy as np
from scipy.special import softmax
import tensorflow as tf
from tqdm.autonotebook import tqdm

from collections import Counter, namedtuple
import logging
import pickle

DEBUG = False
ε = 10E-5
PHI_MIN_LEEWAY = 1.01

if DEBUG:
    from tensorflow.python import debug as tf_debug # pylint: disable=E0401

Estimate = namedtuple("Estimate", "Db, Nb, M, R, P, Nd, phi, avg_M")

class Inference:
    def __init__(self, L, K, N, Q, Gammak, A0,
            learning_rate=0.1,
            threshold=0.05,
            max_iteration=1000,
            learning_steps=10,
            epochs=2,
            verbose=True,
            num_considered_Dbs=128,
            M0_initialization_variance=None,
            use_relative_error=False,
            **model_params):
        self.N = N
        self.L = L
        self.Gammak = Gammak
        self.A0 = A0
        self.K = K
        self.Q = Q
        self.learning_steps = learning_steps
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.threshold = threshold
        self.num_considered_Dbs = num_considered_Dbs
        self.max_iteration = max_iteration
        self.M0_initialization_variance = M0_initialization_variance
        logging.info("Inference initialized with these parameters: %s %s", self.__dict__, model_params)
        
        self.model = tfmodel.build_model(L=L, K=K, N=N,
                                         initial_Gammak=self.Gammak, 
                                         use_Nb_mean=True,
                                         num_Db_values=num_considered_Dbs,
                                         use_relative_error=use_relative_error,
                                         **model_params)
        self.check = None
        self.losses = None

        # global_step = tf.Variable(0., trainable=False)
        # self.learning_rate = tf.compat.v1.train.polynomial_decay(initial_learning_rate, global_step,
            # learning_steps // 2, final_learning_rate, power=1.)
        self.optimizer = tf.train.AdagradOptimizer(
            learning_rate=self.learning_rate).minimize(self.model.loss)


    def fit_one_step(self, M_delta_prev, R_prev, P_prev, P, Nd, Y,
            initialize_M0_logit=None, initialize_phi=None, phi_min=None):
        """Compute or update estimates for a single time step.

        Args:
            M_delta_prev: L x K matrix s.t. M_{t-1} = M_0 + M_delta_prev
            R_prev: L-vector of unsold houses at the last time step.
            P_prev: L-vector of observed average price per location at the previous time step.
            P:  L-vector of observed average price per location at the current time step.
            Nd: L-vector of observed number of deals per location at the current time step.
            Y: K-vector of income of classes at this time step.
            initialize_M0_logit: optional. Initial values for the model variable M0_logit as a
                numpy matrix.
            initialize_phi: optional. Initial value for phi as a float.

        Returns:
            A namedtuple with the estimated values (see outs in tfmodel.py).

        """
        feed_dict = {
            self.model.ins.Y: Y,
            self.model.ins.Q: self.Q,
            self.model.ins.A0: self.A0,
            self.model.ins.Gammak: self.Gammak,
            self.model.ins.R_prev: R_prev,
            self.model.ins.P_prev: P_prev,
            self.model.ins.M_delta_prev: M_delta_prev,
            self.model.ins.given_Nd: Nd,
            self.model.ins.given_P: P,
            self.model.ins.phi_min: phi_min if phi_min is not None else 0.,
        }
        
        self.check = ConvergenceChecker(threshold=self.threshold, verbose=self.verbose,
            max_iteration=self.max_iteration)
        
        with tf.Session() as sess:
            if DEBUG:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            
            sess.run(tf.global_variables_initializer())
                
            if initialize_M0_logit is not None:
                self.model.outs.M0_logit.load(initialize_M0_logit)
                
            if initialize_phi is not None:
                self.model.outs.phi_unbounded.load(initialize_phi)
            
            Y_tilde_min = Y if phi_min is None else (phi_min * Y)
            is_affordable = (
                np.tile(Y_tilde_min[np.newaxis, :], [self.L, 1]) > 
                np.tile(np.array(P_prev)[:, np.newaxis], [1, self.K])
            )
            
            for l in range(self.L):
                if not np.any(is_affordable[l]):
                    raise Exception(
                        f"No class can afford location {l}.\n"
                        f"Its price is {P_prev[l]} and current Y is {Y} (phi_min={phi_min}).")
            
            Db_locs, Db_values = get_Db_per_loc(self.num_considered_Dbs, self.K,
				np.around(Nd).astype(int), is_affordable)
            
            M0, q, loss = None, None, None
            
            while not self.check.is_converged(M0=M0, q=q, loss=loss):
                # E step
                q, log_q = sess.run(
                        [self.model.Db_likelihood, self.model.Db_log_likelihood],
                    feed_dict={
                        **feed_dict,
                        self.model.ins.Db_values: Db_values,
                        self.model.ins.Db_locs: Db_locs,
                    })
                
                if np.any(np.isnan(q)):
                    raise Exception(f"NaN found in q: {q}")
                
                # M step
                feed_dict_with_Db = {
                    **feed_dict,
                    self.model.ins.Db_locs: Db_locs,
                    self.model.ins.Db_values: Db_values,
                    self.model.ins.Db_weights: q,
                }
                
                for step in range(self.learning_steps):
                    sess.run(self.optimizer, feed_dict=feed_dict_with_Db)
                    # debug = sess.run(self.model.debug, feed_dict=feed_dict_with_Db)
                    
                results = sess.run(self.model.outs, feed_dict=feed_dict_with_Db)
                loss = sess.run(self.model.loss, feed_dict=feed_dict_with_Db)
                M0 = results.M0
                
                if any(np.any(np.isnan(v)) for v in results):
                    raise Exception(f'Step {step}. ' + '\n'.join([
                        f"{k} contains NaNs: {v}"
                        for k, v in results._asdict().items() if np.any(np.isnan(v))
                    ]))
                
                if np.any(np.isnan(M0)):
                    raise Exception(f"Step {step}. M0 contains NaNs: {M0}")
                
            mean_Db, mean_P = self.compute_expected_value_per_loc(
                Db_locs, log_q.astype(np.float64),
                Db_values, results.P
            )
                    
            return results, mean_Db, mean_P, loss
    
    def compute_expected_value_per_loc(self, Db_locs, Db_log_weights, *values):
        # Each matrix shape will be transformed from (number of samples, *X) to (number of loc, *X)
        mean_values = [np.full((self.L, ) + value.shape[1:], np.nan) for value in values]
        for l in range(self.L):
            Db_for_location = (Db_locs == l)
            if not np.any(Db_for_location):
                raise Exception(
                   f"Db_locs does not contain location {l}. It contains these: {Counter(Db_locs)}")
            Db_norm_w = softmax(Db_log_weights[Db_for_location])
            if not np.isclose(np.sum(Db_norm_w), 1.):
                raise Exception(f"Overflow in softmax for loc. {l}: sums to {np.sum(Db_norm_w)}")
            for value, mean_value in zip(values, mean_values):
                mean_value[l] = Db_norm_w @ value[Db_for_location] # Like x,xk->k, with other shapes
        return mean_values
    
    def find_phi_min(self, P, Y):
        richest_k = np.argmax(Y[0])
        assert np.all(richest_k == np.argmax(Y, axis=1)), "Income must always be in the same order"
        phi_min = np.max((P.T / Y[:, richest_k]).T) # The phi that works for all t for all locations
        return phi_min * PHI_MIN_LEEWAY
    
    def fit_with_intermediate_results(self, P, Nd, Y, intermediate_basepath=None):
        T, L = P.shape
        K = self.K
        
        if Y.shape == (K, ):
            Y = np.tile(Y, [T, 1])
            
        if self.M0_initialization_variance is None:
            M0_logit = None
        else:
            M0_logit = np.random.randn(L, K) * self.M0_initialization_variance
        
        phi_min = self.find_phi_min(P, Y) if self.model.learnable_phi else None
        
        phi = None
        
        Db_est = np.full((T - 1, L, K), np.nan)
        Nb_est = np.full((T - 1, L, K), np.nan)
        Nd_est = np.full((T - 1, L), np.nan)
        P_est = np.full((T - 1, L), np.nan)
        M_delta_est = np.full((T+1, L, K), np.nan)
        M_delta_est[0] = 0.
        M_est = np.full((T, L, K), np.nan)
        t2M_est = dict() # Estimate of all M (T x L x K) obtained when evaluating at t
        R_est = np.full((T, L), np.nan)
        R_est[0] = np.zeros(L)
        self.losses = np.full(T -1, np.nan)
        
        pbar = tqdm(total=(self.epochs * (T - 1)))
        for epoch in range(self.epochs):
            for t in range(1, T):
                
                # Compute the total Delta = M_{t-1} - M_0.
                # If after some iteration on new t, the delta we decided for previous t ends up
                # being nonsensical, we constrain them to be within (-N, N)
                M_delta_so_far = np.clip(np.sum(M_delta_est[:t], 0), -self.N, self.N)
                
                # Run estimate
                res, last_Db, last_P, last_loss = self.fit_one_step(
                     M_delta_so_far, R_est[t-1], P[t-1], P[t], Nd[t], Y[t],
                     initialize_M0_logit=M0_logit, initialize_phi=phi, phi_min=phi_min)
                     # If M0 or phi were estimated, load their last estimate.
                
                # Gather output
                self.losses[t-1] = last_loss
                Db_est[t-1] = last_Db
                Nb_est[t-1] = res.Nb
                Nd_est[t-1] = res.Nd
                P_est[t-1] = last_P
                M0 = res.M0
                M0_logit = res.M0_logit
                M_delta_est[t] = last_Db - res.Ds
                M_est[0] = res.M0
                M_est[t] = np.maximum(0., res.M0 + M_delta_so_far + last_Db - res.Ds)
                t2M_est[t] = M_est.copy()
                R_est[t] = res.Ns - Nd[t]
                phi = res.phi
                
                if intermediate_basepath is not None:
                    path = f"{intermediate_basepath}/intermediate-e{epoch}-t{t}.pickle"
                    with open(path, 'wb') as f:
                        pickle.dump(
                            dict(
                                res._asdict(),
                                last_Db=last_Db, M_est=M_est, M_delta_est=M_delta_est, R_est=R_est
                            ), f)

                
                pbar.update()
            
            # Compute the estimate for M by using the latest M0 and summing up the changes
            M_est = np.full((T, L, K), np.nan)
            M_est[0] = M0
            for t in range(1, T):
                M_est[t] = np.maximum(0., M_est[t-1] + M_delta_est[t])
            
            # Compute the average estimate for M across all the estimates obtained at each step
            avg_M_est = np.nanmean(list(t2M_est.values()), axis=0)
            
            yield Estimate(Db=Db_est, Nb=Nb_est, M=M_est, R=R_est, P=P_est, Nd=Nd_est, phi=phi,
                avg_M=avg_M_est)
            
        pbar.close()

        
    def fit(self, P, Nd, Y, intermediate_basepath=None):
        results = None
        for results in self.fit_with_intermediate_results(P, Nd, Y,
                                                    intermediate_basepath=intermediate_basepath):
            pass
        return results
        
    def total_loss(self):
        return np.sum(self.losses)
