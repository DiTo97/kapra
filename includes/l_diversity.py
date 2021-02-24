import math
import numpy as np 
import random

def enforce_l_diversity(pattern_dict: dict, A_s_dict: dict, k_group_list: list, l: int, epsilon: int):
    """enforces the l-diversity on the records whose keys are inside A_s_dict

    Parameters
    ----------
    pattern_dict: dict
        dictionary with records keys as keys and pattern representations as values
    
    A_s_dict: dict
        dictionary with records keys as keys and sensitive attribute values as values

    k_group_list: list
        list of k-groups; list elements are record keys
    
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
            if key in k_group:
                PS_R = [k for k in k_group if pattern_dict[k] == pattern_dict[key]]

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
                while A_s_dict[key_ec] in PS_s_values:  # must choose value different from preexisting ones
                    A_s_dict[key_ec] = orig + random.randint(-epsilon, epsilon)
                
                PS_s_values.add(A_s_dict[key_ec])

# just for testing purposes

if __name__=="__main__":
    # dataset to test, p=3, k=6
    pattern_dict = {
        "P1": "abb", "P2": "abb", "P3": "abb", "P4": "abc", "P5": "abc", "P6": "abc", 
        "P7": "bbc", "P8": "bbc", "P9": "bbc", "P10": "abb", "P11": "abb", "P12": "abb",
        "P13": "cba", "P14": "cba", "P15":"cba", "P16":"cbb", "P17":"cbb", "P18":"cbb"
    }

    k_group_list = [
        ["P1", "P2", "P3", "P4", "P5", "P6"], 
        ["P7", "P8", "P9", "P10", "P11", "P12"], 
        ["P13", "P14", "P15", "P16", "P17", "P18"]
    ]

    A_s_dict = {
        "P1": 2, "P2": 3, "P3": 5, 
        "P4": 6, "P5": 1, "P6": 1, 
        "P7": 3, "P8": 3, "P9": 15, 
        "P10": 16, "P11": 16, "P12": 16,
        "P13": 0, "P14": 0, "P15": 1,
        "P16": 2, "P17": 2, "P18": 2
    }

    # l-diversity: we want l=2
    enforce_l_diversity(
        pattern_dict = pattern_dict, 
        A_s_dict = A_s_dict, 
        k_group_list = k_group_list, 
        l = 2, 
        epsilon = 5)
    
    print(A_s_dict)