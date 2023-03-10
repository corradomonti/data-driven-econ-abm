
import numpy as np

import logging


def smape(y_true, y_pred):
    return np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) / len(y_true)


class ConvergenceChecker:
    """ Class to to check if a set of variables reach convergence. """
    def __init__(self, threshold, max_iteration=100000, verbose=False, print_without_check=None):
        self.last_values = None
        self.threshold = threshold
        self.max_iteration = max_iteration
        self.verbose = verbose
        self.iteration = 0
        self.print_without_check = print_without_check or set()
        if self.verbose:
            logging.info("Convergence threshold: %.01E. Max iterations: %s.",
                self.threshold, self.max_iteration)
        
    def is_converged(self, **values):
        """ Returns True if the mean of the differences w.r.t. the last call of this
            method is less than the threshold _for each_ of the given values.
        """
        
        values_to_check = {k: v for k, v in values.items() if k not in self.print_without_check}
        
        if any(v is None for v in values_to_check.values()):
            return False
        
        if self.last_values is None:
            self.last_values = values_to_check
            return False
        
        assert set(self.last_values.keys()) == set(values_to_check.keys())
            
        diffs = {k: 
            np.mean(np.abs(self.last_values[k] - v).flatten()) for k, v in values_to_check.items()}
        if self.verbose:
            stats = '\t'.join("%s: %.01E" % (k, diffs[k]) for k in sorted(diffs))
            extra_stats = '\t'.join("%s: %s" % (k, values.get(k)) for k in self.print_without_check)
            logging.info("Iteration %d.\tDiffs:\t%s.\t%s", self.iteration, stats, extra_stats)
        self.last_values = values_to_check
        self.iteration += 1
        return (all(v < self.threshold for v in diffs.values()) 
            or self.iteration > self.max_iteration)
        

def partitions(length, total_sum, step=1):
    """ Iterates all positive integers vector of length `length` that sum to `total_sum`,
        by moving in increments of `step`.
    """
    if length == 1:
        yield (total_sum,)
    else:
        for v in range(0, total_sum + 1, step):
            for permutation in partitions(length - 1, total_sum - v, step):
                yield (v,) + permutation

assert set(partitions(3, 2)) == {(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)}

def homophily_matrix(K, k_delta, dtype=np.float32):
    """ Returns a matrix representing homophily between K classes, where each class k is attracted
        from other classes at distance at most k_delta.
    """
    H = np.array([
        [np.abs(i - j) <= k_delta for i in range(K)] for j in range(K)
    ], dtype)
    
    # for i in range(K): # H[k,k] increases s.t. the sum of H[k] is equal for all classes k
    #     while np.sum(H[i]) < (1 + 2 * k_delta):
    #         H[i, i] += 1
    return H
    
