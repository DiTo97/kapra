from dataset_anonymized import DatasetAnonymized
import math
import numpy as np 
import random


def get_time_series_envelope(pattern_anonymized_data: list, key: str):
    """given anonymized dataset T_star and  a time-series Q key, it returns PS(Q)
    """
    # inside the right pattern group, whose serieses have same anonymization envelope
    # as Q? Those are PS(Q)
    for p_group in pattern_anonymized_data:
        for node in p_group:
            if key in node.group.keys():
                return list(node.group.keys())


def get_equivalence_class(v_i: int, timeSeriesEnvelope: list, vi_dict: dict):
    """given an attribute value v_i, finds its equivalence class.

    Parameters
    ----------
    v_i : int
        sensitive data value to be matched against
    
    timeSeriesEnvelope : list
        PS(Q) (as list of keys)

    vi_dict : dict
        dictionary containing row id as key and sensitive attribute as value
    """
    ecs = []
    for key in timeSeriesEnvelope:
        if key in vi_dict and vi_dict[key] == v_i:
            ecs.append(key)
    return ecs

def get_equivalence_classes(vi_dict: dict, pattern_anonymized_data: list):
    """given a time series dataset, yields the equivalence classes
    as a dictionary of lists (being the equivalence classes)
    
    Parameters
    ----------
    vi_dict : dict
        dictionary containing row id as key and sensitive attribute as value
    """
    d = vi_dict.copy()
    # set is better than list here since we'll delete lots of elements
    keyset = set(d.keys())
    ecs = {}
    for key in keyset.copy():
        ec = get_equivalence_class(
            vi_dict[key], 
            get_time_series_envelope(pattern_anonymized_data, key), 
            d
        )
        ecs[vi_dict[key]] = ec

        for el in ec: 
            keyset.remove(el)
            del d[el]
    return ecs


def enforce_l_diversity(pattern_anonymized_data: list, l: int, vi_dict: dict, epsilon: int = 1):
    """
    enforces l-diversity.

    Parameters
    ----------
    pattern_anonymized_data : list
        the dataset anonymized by pattern

    l : int
        value for l-diversity

    vi_dict : dict
        dictionary containing row id as key and sensitive attribute as value

    epsilon : int
        how much to perturbate sensitive data
    """

    ecs = get_equivalence_classes(vi_dict, pattern_anonymized_data)
    for key in vi_dict:
        ts_envelopes = get_time_series_envelope(pattern_anonymized_data, key)

        l_val = len(ecs[vi_dict[key]])/len(ts_envelopes)
        if l_val <= 1/l: continue  # l-diversity is satisfied

        x_i = len(ecs[vi_dict[key]]) - math.floor(len(ts_envelopes)/l)
        records = np.random.default_rng().choice(ecs[vi_dict[key]], size=x_i, replace=False)

        for r in records:
            orig_value = vi_dict[r]
            # will the change impact l-diversity?
            while vi_dict[r] in ecs:  # ecs has as keys all different sensitive attributes
                vi_dict[r] = orig_value + random.randint(-epsilon, epsilon)
