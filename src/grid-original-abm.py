
import mlflow
import numpy as np
import scipy.stats
import sklearn.metrics
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
np.set_printoptions(suppress=True, precision=2, threshold=2000)

import em
import plots
from utils import smape

import inspect
import logging
import os
import pickle
import sys
import traceback

DEBUG = False

N = 1000
Q = 500
K = 3
L = 5
T = 20
T_forecast = 5
IOTA = 1.
ALPHA = 0.1
TAU = 1.
BETA = 0.5
NU = 0.1
A0 = np.ones(L)
Gammak = np.array([0.5, 0.4, 0.1])
Y = np.array([10, 50, 90])

def log_arguments(func):
    func_signature = inspect.signature(func)
    def decorated_func(*args, **kwargs):
        signature_binded = func_signature.bind(*args, **kwargs)
        signature_binded.apply_defaults()
        mlflow.log_params(signature_binded.arguments)
        return func(*args, **kwargs)
    return decorated_func
    
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
    
@log_arguments
def run_experiment(trace_num, delta,
        num_considered_Dbs, learning_rate, learning_steps,
        threshold, M0_initialization_variance,
        price_error_stddev, num_deals_error_stddev,
        max_iteration, epochs, num_restarts, use_relative_error, seed
    ):
    
    mlflow.log_param('N', N)
    mlflow.log_param('Q', Q)
    mlflow.log_param('Gammak', Gammak)
    
    np.random.seed(seed)
    
    file_logger = logging.FileHandler("experiment.log")
    file_logger.setLevel(logging.INFO)
    file_logger.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_logger)
    logging.info('==============================New Experiment==================================')
    
    P = np.genfromtxt(
        f"../data/old_abm/traces/prices{trace_num}.tsv")
    Nd = np.genfromtxt(
        f"../data/old_abm/traces/transactions{trace_num}.tsv")
    M = np.genfromtxt(
        f"../data/old_abm/traces/distribution-agents{trace_num}.tsv")
    Db = np.genfromtxt(
        f"../data/old_abm/traces/buyers{trace_num}.tsv")
    
    assert P.shape == (T + T_forecast, L)
    assert Nd.shape == (T + T_forecast, L)
    assert M.shape == (T + T_forecast, L * K)
    assert Db.shape == (T + T_forecast, L * K)
    
    M = M.reshape(T + T_forecast, L, K)[:T]
    Db = Db.reshape(T + T_forecast, L, K)[:T]
    P = P[:T]
    Nd = Nd[:T]
    
    best_loss = np.inf
    
    for i_restart in range(num_restarts):
        learner = em.Inference(
            L=L, K=K, N=N, Q=Q, alpha=ALPHA,
            tau=TAU, beta=BETA, nu=NU, delta=delta,
            Gammak=Gammak,
            A0=A0,
            num_considered_Dbs=num_considered_Dbs,
            M0_initialization_variance=M0_initialization_variance,
            learning_rate=learning_rate,
            learning_steps=learning_steps,
            threshold=threshold,
            max_iteration=max_iteration,
            verbose=True,
            epochs=epochs,
            homophily=None,
            price_error_stddev=price_error_stddev,
            num_deals_error_stddev=num_deals_error_stddev,
            learnable_phi=False,
            use_relative_error=use_relative_error,
        )

        est = learner.fit(P, Nd, Y)
        
        np.savetxt(f'M0_est_restart{i_restart}.txt', est.M[0], '%.0f')
        mlflow.log_artifact(f'M0_est_restart{i_restart}.txt')
        
        if learner.total_loss() < best_loss:
            best_estimate = est
            best_loss = learner.total_loss()
    
    est = best_estimate
    mlflow.log_metric('phi_estimated', est.phi)
        
    for y_true, y_pred, name in [
            (M[0], est.M[0], 'M0'),
            (M, est.M, 'M'),
            (M, est.avg_M, 'avg_M'),
            (M[0], est.avg_M[0], 'avg_M0'),
            (P[1:], est.P, 'P'),
            (Nd[1:], est.Nd, 'Nd'),
            (Db[1:], est.Db, 'Db'),
            (Db[:-1], est.Db, 'Db_m1'),
    ]:
        assert y_true.shape == y_pred.shape, f"{name} mismatch: {y_true.shape} != {y_pred.shape}"
        
        y_true[~np.isfinite(y_true)] = 0.
        y_pred[~np.isfinite(y_pred)] = 0.
        
        pearson_value = scipy.stats.pearsonr(y_true.flatten(), y_pred.flatten())[0]
        mlflow.log_metric(f'pearson_{name}', pearson_value)
        # mlflow.log_metric(f'pearson_{name}_epoch_{epoch}', pearson_value)
        smape_value = smape(y_true.flatten(), y_pred.flatten())
        mlflow.log_metric(f'smape_{name}', smape_value)

        for metric_name in ([
            'explained_variance_score',
            'max_error',
            'mean_absolute_error',
            'mean_squared_error',
            'r2_score',
        ]):
            metric_fun = getattr(sklearn.metrics, metric_name)
            metric_value = metric_fun(y_true.flatten(), y_pred.flatten())
            mlflow.log_metric(f'{metric_name}_{name}', metric_value)
            # mlflow.log_metric(f'{metric_name}_{name}_epoch_{epoch}', metric_value)
        
        mlflow.log_metric('total_loss', learner.total_loss())
    
    np.savetxt('M0_est.txt', est.M[0], '%.0f')
    mlflow.log_artifact('M0_est.txt')
    np.savetxt('MT_est.txt', est.M[-1], '%.0f')
    mlflow.log_artifact('MT_est.txt')
    
    with open("estimate.pickle", 'wb') as f:
        pickle.dump(est, f)
    mlflow.log_artifact("estimate.pickle")
    
    logging.info('================================End Experiment================================')
    logging.getLogger().removeHandler(file_logger)
    mlflow.log_artifact(file_logger.baseFilename)
    os.remove(file_logger.baseFilename)
    
    plots.make_plots(M=M, M_est=est.M, P=P, P_est=est.P, Y=Y)
    
    mlflow.log_metric('final_est_num_agents', np.sum(est.M[-1]))
    mlflow.log_metric('final_num_agents', np.sum(M[-1]))

def main():
    experiment_id = "main-experiments-original-abm"
    mlflow.set_experiment(experiment_id)
    
    for M0_initialization_variance in (1, ):
        for num_considered_Dbs in (256, ):
            for threshold in (0.05, ):
                for learning_steps in (4, ):
                    for learning_rate in (0.001, ):
                        for num_deals_error_stddev in (1., ):
                            for delta in (0.0625, ):
                                for seed in range(43, 53):
                                    for use_relative_error in (False, ):
                                        for trace_num in range(10, 21):
                                            with mlflow.start_run():
                                                try:
                                                    run_experiment(
                                                        trace_num=trace_num,
                                                        delta=delta,
                                                        num_considered_Dbs=num_considered_Dbs,
                                                        learning_rate=learning_rate,
                                                        learning_steps=learning_steps,
                                                        threshold=threshold, 
                                                        M0_initialization_variance=M0_initialization_variance,
                                                        price_error_stddev=1., 
                                                        num_deals_error_stddev=num_deals_error_stddev,
                                                        max_iteration=100,
                                                        epochs=5, 
                                                        num_restarts=1,
                                                        use_relative_error=use_relative_error,
                                                        seed=seed,
                                                    )
                                                except Exception: # pylint: disable=broad-except
                                                    mlflow.set_tag('crashed', True)
                                                    with open("exception.txt", 'w') as _f:
                                                        traceback.print_exc(file=_f)
                                                    mlflow.log_artifact("exception.txt")
                                                    raise
    
                                                if not os.path.exists('../data/oabm-experiments/'):
                                                    os.makedirs('../data/oabm-experiments/')
                                                mlflow.search_runs().to_csv(
                                                    f"../data/oabm-experiments/{experiment_id}.csv", index=False)

if __name__ == '__main__':
    main()
