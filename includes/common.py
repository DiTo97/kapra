import os
import random
import sys
import math
import numpy as np

ROUNDS = 6 # # of NCP maximization rounds. By up to 6 rounds,
           # we can achieve more than 98.75% of the maximal penalty

MAX_LEVEL = 5 # Maximum # of different chars in SAX pattern representations

# Custom imports #
from .metric import instant_value_loss
from .metric import normalized_certainty_penalty

from .node import Node

def find_tuple_with_max_ncp(base, T, key, T_max_vals, T_min_vals):
    """
    Scan through the whole table T, and find the i-th tuple that maximizes NCP(base, i).
    
    Parameters
    ----------
    :param base: list of int
        Tuple to compare T's tuples against

    :param key: T
        the table

    :param key: int
        Unique Id of `base`

    :param T_max_vals
        the maximums
    
    :param T_min_vals
        the minimums
        
    Returns
    -------
    :return best: int
        Unique Id of the found tuple
    """

    max_ncp = 0
    best = None

    for k in T.keys():
        if k != key:
            ncp = normalized_certainty_penalty([base, T[k]],
                    T_max_vals, T_min_vals)

            if ncp >= max_ncp: # Update the best tuple Id
                max_ncp = ncp
                best = k

    return best

def find_tuple_with_max_vl(base, T, key):
    """
    Scan through the whole table T, and find the i-th tuple that maximizes VL(base, i).

    Returns
    -------
    :return best: int
        Unique Id of the found tuple
    """

    max_vl = 0
    best = None

    for k in T.keys():
        if k != key:
            vl = instant_value_loss([base, T[k]])

            if vl >= max_vl: # Update the best tuple Id
                max_vl = vl
                best = k

    return best
    
def find_group_with_min_vl(group_to_search=None, group_to_merge=dict(), index_ignored=list()):
    min_p_group = {"group" : dict(), "index" : None, "vl" : float("inf")} 
    for index, group in enumerate(group_to_search):
        if index not in index_ignored: 
            vl = instant_value_loss(list(group.values()) + list(group_to_merge.values()))
            if vl < min_p_group["vl"]:
                min_p_group["vl"] = vl
                min_p_group["group"] = group
                min_p_group["index"] = index

    return min_p_group["group"], min_p_group["index"]

def top_down_greedy_clustering(algorithm, T, size, T_clustered,
        T_structure, label='o', T_max_vals=None, T_min_vals=None):
    """
    Top down greedy search implementation, from Xu et al. 2006,
    Utility-based Anonymization for Privacy Preservation with Less Information Loss, 4.2

    It mimics the construction of a binary tree with a number of separate list/dict structures. At each clustering level the data is split
    in two smaller groups, each minimizing the intra-NCP (naive) or -VL (KAPRA) among its records. Each bipartite group is marked
    with a unique label, which extends the label of its larger parent group, in order to track its path from root to tip.

    Parameters
    ----------
    :param algorithm: str
        (k, P)-anonymity implementation: naive or KAPRA

    :param T: dict of list of int
        Dict of time-series records on QI attributes

    :param size: int
        Cluster size

    :param T_clustered: list of dict of list of int
        List of `size`-large clustered groups from `T`

    :param T_structure: list of str
        List of unique alphabetic labels identifying clustered groups in `T_clustered`

    :param label: str - 'o'
        Alphabetic label mapping the current clustering level

    :param T_max_vals: list of int - None
        List of max values for each QI attribute

    :param T_min_vals: list of int - None
        List of min values for each QI attribute
    """

    # If there are less than 2*size records in T, there is no way
    # to produce two valid cuts >= size. The recursion can then stop.
    if len(T) < 2*size:
        T_clustered.append(T)
        T_structure.append(label)
        return

    ids = list(T.keys())

    # 1. Initialize groups via a NCP maximization-based heuristic
    group_u = dict()
    group_v = dict()

    seed = ids[random.randint(0, len(ids) - 1)] # Draw a random row Id
    group_u[seed] = T[seed]

    old = seed # Last visited record

    # to avoid this row to end up in two different groups
    del T[seed]
    ids.remove(seed)

    # 1.a Fill the two groups alternately for # of ROUNDS
    # while maximiziming the respective NCP (naive) or IVL (KAPRA) metric
    rounds = ROUNDS if len(T) >= ROUNDS else len(T)

    for rnd in range(rounds):
        if rnd % 2 == 0:
            source = group_u
            target = group_v
        else:
            source = group_v
            target = group_u

        if algorithm == 'naive':
            r = find_tuple_with_max_ncp(source[old], T, old, \
                    T_max_vals, T_min_vals)
        elif algorithm == 'kapra':
            r = find_tuple_with_max_vl(source[old], T, old)

        target[r] = T[r]
        old = r

        # Update data structures
        del T[r]
        ids.remove(r)

    # 1.b Assign each record to the group with lower NCP
    random.shuffle(ids) # Shuffle leftover Ids

    for i in ids:
        row = T[i]

        # Copy values to check what would happen
        # if row was added to either one separately
        group_u_vals = list(group_u.values())
        group_v_vals = list(group_v.values())

        group_u_vals.append(row)
        group_v_vals.append(row)

        if algorithm == 'naive':
            metric_u = normalized_certainty_penalty(group_u_vals, \
                    T_max_vals, T_min_vals)
            metric_v = normalized_certainty_penalty(group_v_vals, \
                    T_max_vals, T_min_vals)
        elif algorithm == 'kapra':
            metric_u = instant_value_loss(group_u_vals)
            metric_v = instant_value_loss(group_v_vals)

        if metric_v < metric_u:
            group_v[i] = row
            del group_u_vals[-1]
        else:
            group_u[i] = row
            del group_v_vals[-1]

        del T[i]

    # 2. Iterate recursively, or store groups if base case
    if len(group_u) >= size:
        top_down_greedy_clustering(algorithm, group_u, size, T_clustered, \
                T_structure, label + 'a', T_max_vals, T_min_vals) # Extend label with 'a'
    else:
        T_clustered.append(group_u)
        T_structure.append(label + 'a')

    if len(group_v) >= size:
        top_down_greedy_clustering(algorithm, group_v, size, T_clustered, \
                T_structure, label + 'b', T_max_vals, T_min_vals) # Extend label with 'b'
    else:
        T_clustered.append(group_v)
        T_structure.append(label + 'b')


def postprocessing(algorithm, size, T_clustered, T_structure,
        T_postprocessed, T_max_vals=None, T_min_vals=None):
    """
    Top down greedy search postprocessing, from Xu et al. 2006,
    Utility-based Anonymization for Privacy Preservation with Less Information Loss, 4.2 bottom

    It prevents final groups from being smaller than `size`. Indeed `top_down_greedy_clustering()` may produce some sub-`size` groups.
    For each such group G, it will be checked whether it is more beneficial, in terms of NPC (naive) or VL (KAPRA), to merge it:
        - with its nearest neighbour, that is, the group with the most similar label in `T_structure`;
        - with the group at least large (2*`size` - |G|) that minimizes the metric.

    Parameters
    ----------
    :param T_postprocessed: list of dict of list of int
        List of good groups merged from `T_clustered`
    """

    idxs_merged = list()      # Already visited groups
    groups_merged = list()    # Resulting merged groups
    structure_merged = list() # Updated group labels


    # 1. Find the two candidate groups
    # print("T clustered is ", T_clustered)
    """ print("T_structure is ", T_structure) """
    for idx, bad_group in enumerate(T_clustered):
        bad_g_size = len(bad_group)
        if bad_g_size < size: # For any bad group
            bad_group_vals = list(bad_group.values())
            """ print("idx is ", idx) """
            label = T_structure[idx]

            # 1.a Find its nearest neighbour (NN) - 1st candidate group

            # Search the group's NN as the one
            # with the most similar label
            idx_nn = -1
            found_nn = False
            metric_nn = float('inf')

            # print("T_structure is ", T_structure)
            # print("Bad group label is ", label)
            for other_idx, other_label in enumerate(T_structure):
                # Per label construction, if the two labels
                # bar the last char are equal, it means the two groups
                # come from the same parent; hence they are the respective NN
                # print("other label ", other_label)
                if label[:-1] == other_label[:-1]: 
                    if idx == other_idx:
                        continue

                    # If the group hasn't already been merged
                    # with another group, mark it as a valid NN
                    if other_idx not in idxs_merged:
                        found_nn = True
                        idx_nn = other_idx
                        break
            # print("bad group index ", idx)
            # print("indexes merged ", idxs_merged)

            merge_with_other_group = False
            if found_nn:
                group_nn = T_clustered[idx_nn]
            elif idx_nn !=idx:
                if idx - 1 > 0:
                    idx_nn = idx - 1
                elif idx + 1 < len(T_structure) - 1:
                    idx_nn = idx + 1 
                group_nn = T_clustered[idx_nn]
                merge_with_other_group = True

            if found_nn or merge_with_other_group:
                group_merged_nn = bad_group_vals

                
                group_merged_nn = group_merged_nn  \
                            + list(group_nn.values())
                
                if algorithm == 'naive':
                    metric_nn = normalized_certainty_penalty(group_merged_nn,
                            T_max_vals, T_min_vals)
                elif algorithm == 'kapra':
                    metric_nn = instant_value_loss(group_merged_nn)

                    # Redefine group_merged_nn as dict
                group_merged_nn = dict()
                group_merged_nn.update(bad_group)
                group_merged_nn.update(group_nn)

            # 1.b Find the most appropriate large group (>= 2*size -|G|) - 2nd candidate group
            metric_large_g = float('inf')
            idx_large_g = -1

            for other_idx, other_group in enumerate(T_clustered):
                # If the group is large enough
                if len(other_group) >= 2*size - bad_g_size: # 2*size - |G|
                    # print("dentro if large group metric")
                    if other_idx not in idxs_merged:
                        group_merged_large_g = bad_group.copy()
                        group_large_g_vals = list(group_merged_large_g.values())

                        # Select the size - |G| records from the large group that minimize
                        # the intra-NCP or VL metric with the original group
                        for j in range(size - bad_g_size): # size - |G|
                            tmp_metric = float('inf')

                            best_record = {}
                            best_row = []

                            # Select the best record to merge
                            # at the j-th iteration
                            for ridx, row in other_group.items():
                                if ridx not in group_merged_large_g.keys():
                                    if algorithm == 'naive':
                                        metric = normalized_certainty_penalty(group_large_g_vals + [ row ],
                                                T_max_vals, T_min_vals)
                                    elif algorithm == 'kapra':
                                        metric = instant_value_loss(group_large_g_vals + [ row ])

                                    if metric < tmp_metric: # Update min metric
                                        best_record = { ridx : row }
                                        tmp_metric = metric
                                        best_row = row
            
                            group_merged_large_g.update(best_record)
                            group_large_g_vals.append(best_row)

                        # Check if the current candidate large group
                        # is better than any previous ones
                        if tmp_metric < metric_large_g:
                            metric_large_g = tmp_metric
                            idx_large_g = other_idx

                            # Isolate the records that are kept from
                            # the original (2*size - |G|) large group
                            leftover_group_large_g = { k : val for (k, val)
                                    in other_group.items()
                                    if k not in group_merged_large_g.keys() }
            # print("group_merged_large_g \n\n", group_merged_large_g)
            """ print("Metric nn: ", str(metric_nn))
            print("Metric large group: ", str(metric_large_g))
            print("Found nn: ", str(found_nn))
            # print("group merged nn ", group_merged_nn)
            print("Bad group label: ", label)
            print("Bad group: ", bad_group) """
            # 1.c Choose which of the two candidate
            # groups is best to merge with
            """ if math.isinf(metric_nn) and math.isinf(metric_large_g):
                continue  """
            if metric_nn < metric_large_g: 
                idxs_merged.append(idx_nn)
                groups_merged.append(group_merged_nn)
                structure_merged.append(label[:-1]) 
                # print("NN CASE")
            else:
                # print("dentro else")
                idxs_merged.append(idx_large_g)
                # Add both groups to merge
                groups_merged.append(group_merged_large_g)
                groups_merged.append(leftover_group_large_g)
                # print("LARGE GROUP CASE")
                structure_merged.append('') # Add empty label for new group

            # Add the currently processed group Id
            # to already visited groups Ids
            idxs_merged.append(idx)

    # 2. Re-assest outer data structures
    T_clustered = [ group for (idx, group)
            in enumerate(T_clustered)
            if idx not in idxs_merged ]
    T_clustered += groups_merged 

    T_structure = [ label for (idx, label)
            in enumerate(T_structure)
            if idx not in idxs_merged]
    T_structure += structure_merged

    T_postprocessed += T_clustered

    # 3. Check if there are any more bad groups
    bad_groups_cnt = 0

    for group in T_clustered:
        if len(group) < size:
            bad_groups_cnt +=1

    # print("Number of bad groups before a possible recursive call: ", str(bad_groups_cnt))

# def postprocessing(algorithm, size, T_clustered, T_structure, T_postprocessed, T_max_vals=None, T_min_vals=None):
    if bad_groups_cnt > 0: # Call recursively if any left
        postprocessing(algorithm, size, T_clustered, T_structure,
                T_postprocessed, T_max_vals, T_min_vals)

def create_tree(algorithm, T, PR, P_value, paa_value, max_level=MAX_LEVEL):
    """
    Split a group of records into sub-groups of at least `P_value` records with the same pattern. This procedure applies to both naive
    and KAPRA (k, P)-anonymity, starting from a k-group or from the whole time series data, respectively.

    Parameters
    ----------
    :param PR: dict
        Dict of per-record pattern representations from `T`
    """
    # P-groups leaf nodes
    bad_leaf_nodes  = list()
    good_leaf_nodes = list()

    node = Node(level=1, group=T, paa_value=paa_value)
    node.start_splitting(P_value, max_level, good_leaf_nodes, bad_leaf_nodes)

    suppressed_nodes = list()
        
    if len(bad_leaf_nodes) > 0:
        if algorithm == 'naive':
            Node.postprocessing(good_leaf_nodes, bad_leaf_nodes)
        elif algorithm == 'kapra':
            Node.recycle_bad_leaves(P_value, good_leaf_nodes,
                    bad_leaf_nodes, suppressed_nodes, paa_value)

    suppressed_groups = list()
    P_groups = list()

    for node in suppressed_nodes:
        suppressed_groups.append(node.group)

    for node in good_leaf_nodes:
        P_groups.append(node.group)
        pr = node.pattern_representation

        for key in node.group:
            PR[key] = pr

    return P_groups, suppressed_groups
