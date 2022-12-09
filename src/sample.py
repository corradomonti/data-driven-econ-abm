import utils

import numpy as np
from scipy.special import binom

import itertools
import logging

def count_how_many_possible_Dbs_per_loc(K, Nd_loc, step=1):
    return int(binom(Nd_loc // step + K - 1, Nd_loc // step))

def get_num_Dbs_per_loc(how_many, Nd, K_per_location):
    num_possibilities_per_loc = np.array([
        count_how_many_possible_Dbs_per_loc(K_per_location[l], Nd[l])
        for l in range(len(Nd))
    ])
    sample_rate = np.sum(num_possibilities_per_loc) / (how_many - len(Nd))
    num_Db_per_loc = (num_possibilities_per_loc / sample_rate).astype(np.int) + 1
    while np.sum(num_Db_per_loc) < how_many:
        num_Db_per_loc[np.argmin(num_Db_per_loc)] += 1
    return num_Db_per_loc

def get_Db_per_loc(how_many, K, Nd, is_affordable):
    """ Returns two parallel tensors: the first is a vector where each index represent a location l
        and the second tensor a possible value of Db at location l.
    """
    # How many classes can afford each location.
    K_per_location = np.sum(is_affordable, axis=1)
    # Count how many we want per location.
    num_Db_per_loc = get_num_Dbs_per_loc(how_many, Nd, K_per_location)
    
    locs = []
    Dbs = []
    
    for l, how_many_in_l in enumerate(num_Db_per_loc):
        # For each location, find the right step to take how_many_in_l samples.
        for s in itertools.count():
            logging.debug(f"Trying step for location {l}: 2^{s}")
            t = count_how_many_possible_Dbs_per_loc(K_per_location[l], Nd[l], 2 ** s)
            logging.debug(f"How many: {t}. Required: {how_many_in_l}.")
            if t <= how_many_in_l:
                break
        step = max(1,2 ** (s - 1)) # pylint: disable=W0631
        # Take too many of them considering K as the number of classes that can afford it...
        affordable_Dbs = np.array(list(utils.partitions(K_per_location[l], Nd[l], step=step)))
        # ...let's map them to the original K...
        all_Dbs = np.zeros((len(affordable_Dbs), K))
        all_Dbs[:, is_affordable[l]] = affordable_Dbs
        # ...and subsample them.
        indexes = np.random.choice(np.arange(len(all_Dbs)), size=how_many_in_l, replace=True)
        
        locs.append(np.full(how_many_in_l, l))
        Dbs.append(all_Dbs[indexes])
        logging.debug(f"Sampled Dbs for location {l}.")
    
    locs = np.hstack(locs)
    Dbs = np.vstack(Dbs)
    
    assert locs.shape == (how_many, )
    assert Dbs.shape == (how_many, K)
    
    return locs, Dbs
