import numpy as np
import tensorflow as tf

def softmax_by_index(logit, indexes, max_index=None):
    """ Given a vector `logit` and a parallel vector of `indexes`, returns the softmax of `logit`
        as if it was applied separately with elements with same index in `indexes`.
        `max_index` is the range of `indexes` (the largest index must not be beyond `max_index`).
        
        For example, if `logit` is log([15., 5., 2., 7., 2.,]) and `indexes` is [0, 0, 1, 2, 1],
        it returns log([0.75, 0.25, 0.5, 1., 0.5 ]).
    """
    log_norms = tf.stack([
        tf.math.reduce_logsumexp(logit[tf.equal(indexes, i)])
        for i in range(max_index)
    ])
    return logit - tf.gather(log_norms, indexes)
    
if __name__ == '__main__':
    tlogit = tf.log(tf.constant(np.array([
        15., 5., 2., 10., 2.,
    ])))
    tindexes = tf.constant(np.array([
        0,   0,  1,   2,  1,
    ]))
    with tf.Session() as sess:
        np.testing.assert_almost_equal(
            np.exp(softmax_by_index(tlogit, tindexes, 3).eval()),
              np.array([0.75, 0.25, 0.5, 1., 0.5 ])
        )
    
    # Check the normalization when a value is -10E6
    tlogit_with_minus_infty = tf.constant(np.array([
        10., 10., -10E6, 2., 2., 2.,
    ]))
    tindexes_with_minus_infty = tf.constant(np.array([
        1,   1,  1,      0,  0,  0,
    ]))
    with tf.Session() as sess:
        np.testing.assert_almost_equal(
            np.exp(softmax_by_index(tlogit_with_minus_infty, tindexes_with_minus_infty, 2).eval()),
              np.array([0.5, 0.5, 0., 1/3, 1/3, 1/3 ])
        )
