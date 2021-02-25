import math
import numpy as np 
import random

from loguru import logger

def enforce_l_diversity(pattern_dict: dict, A_s_dict: dict, k_group_list: list, l: int, epsilon: int = 3):
    """enforces the l-diversity on the records whose keys are inside A_s_dict

    Parameters
    ----------
    pattern_dict: dict
        dictionary with records keys as keys and pattern representations as values
    
    A_s_dict: dict
        dictionary with records keys as keys and sensitive attribute values as values

    k_group_list: list
        list of k-groups
    
    l: int
        l-value for l-diversity

    epsilon: int
        how much to potentially perturbate data (data will be perturbed of a value in range [-epsilon, epsilon])
    """
    PS_R = None
    keyset = set()

    for key in A_s_dict:  # loop over record keys. A_s_dict[key] is sensitive data of that record
        if key in keyset: continue  # already dealt with that record
        
        keyset.add(key)

        # find PS(Q)
        for k_group in k_group_list:
            # derive the right p-group
            if key in k_group.keys():
                PS_R = [k for k in k_group if pattern_dict[k] == pattern_dict[key]]
                break

        # find equivalence class, i.e. records in PS_R having same sensitive attribute
        EC_v = [k for k in PS_R if A_s_dict[k] == A_s_dict[key]]
        keyset.update(EC_v)

        PS_s_values = {A_s_dict[k] for k in PS_R}
        # some tuples can be suppressed after (k,p)-anonymity, so PS_R and EC_v might be
        # empty - in which case we'll just skip this remaining step.
        if PS_R and EC_v:
            # l-diversity is satisfied, no need to take action
            if len(EC_v) / len(PS_R) <= 1/l: continue

            # data needs to be perturbed.
            x_i = len(EC_v) - math.floor(len(PS_R)/l)

            for key_ec in np.random.default_rng().choice(EC_v, size=x_i, replace=False):
                orig = A_s_dict[key_ec]

                noises = [ x - epsilon for x in range(2*epsilon + 1) ]
                random.shuffle(noises)

                perturbated = False

                for noise in noises:
                    A_s_dict[key_ec] = orig + noise

                    if A_s_dict[key_ec] not in PS_s_values:
                        perturbated = True
                        break
                
                if perturbated:
                    PS_s_values.add(A_s_dict[key_ec])
                else:
                    """
                    No valid perturbative noise was found, where valid means that it did not previosuly exist inside the P-group.
                    In order not to falsify the frequency of the existing sensitive attributes inside the P-group, we discard the
                    random procedure, and operate with an iterative +/- 1 increment of the perturbative noise.

                    By doing so we favour the satisfaction of the l-diversity constraint at the expenses of larger information loss;
                    hence we improved the privacy capability, but degraded the utility in return.
                    """
                    round = 1
                    increment = 1
                    while True:
                        # Iteratively extend the +/- boundary
                        # of the perturbative noise
                        pos_noise = epsilon  + (increment*round)
                        neg_noise = -epsilon - (increment*round)

                        noises = [pos_noise, neg_noise]

                        perturbated = False

                        for noise in noises:
                            A_s_dict[key_ec] = orig + noise

                            if A_s_dict[key_ec] not in PS_s_values:
                                perturbated = True
                                break

                        if perturbated:
                            PS_s_values.add(A_s_dict[key_ec])
                            logger.error('Perturbated record ' + str(key_ec)
                                    + ' only at round #' + str(round))
                            break
                        else:
                            round += 1
        