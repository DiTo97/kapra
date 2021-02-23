import os
import random
import sys

import numpy as np

ROUNDS = 6 # # of NCP maximization rounds. By up to 6 rounds,
           # we can achieve more than 98:75% of the maximal penalty

# Custom imports #
from .metric import instant_value_loss
from .metric import normalized_certainty_penalty

def find_tuple_with_max_ncp(base, T, key, T_max_vals, T_min_vals):
    """
    Scan through the whole table T, and find the i-th tuple that maximizes NCP(base, i).
    
    Parameters
    ----------
    :param base: list of int
        Tuple to compare T's tuples against

    :param key: int
        Unique Id of `base`

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

    # 1.a Fill the two groups alternately for # of ROUNDS
    # while maximiziming the respective NCP (naive) or IVL (KAPRA) metric
    for round in ROUNDS:
        if round % 2 == 0:
            source = group_u
            target = group_v
        else:
            source = group_v
            target = group_u

        if algorithm == 'naive':
            r = find_tuple_with_max_ncp(source[old], T, old,
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
            metric_u = normalized_certainty_penalty(group_u_vals,
                    T_max_vals, T_min_vals)
            metric_v = normalized_certainty_penalty(group_v_vals,
                    T_max_vals, T_min_vals)
        elif algorithm == 'kapra':
            metric_u = instant_value_loss(group_u_vals)
            metric_v = instant_value_loss(group_v_vals)

        if metric_v < metric_u:
            group_v[i] = row
        else:
            group_u[i] = row

        del T[i]

    # 2. Iterate recursively, or store groups if base case
    if len(group_u) > size:
        top_down_greedy_clustering(algorithm, group_u, size, T_clustered,
                T_structure, label + 'a', T_max_vals, T_min_vals) # Extend label with 'a'
    else:
        T_clustered.append(group_u)
        T_structure.append(label)

    if len(group_v) > size:
        top_down_greedy_clustering(algorithm, group_v, size, T_clustered,
                T_structure, label + 'b', T_max_vals, T_min_vals) # Extend label with 'b'
    else:
        T_clustered.append(group_v)
        T_structure.append(label)

def postprocessing(algorithm, size, T_clustered, T_structure,
        T_postprocessed, T_max_vals=None, T_min_vals=None):
    """
    Top down greedy search postprocessing, from Xu et al. 2006,
    Utility-based Anonymization for Privacy Preservation with Less Information Loss, 4.2 bottom

    It prevents final groups from being smaller than `size`. Indeed `top_down_greedy_clustering()` may produce some sub-`size` groups.
    For each of those groups, G, it will be checked whether it is more beneficial, in terms of NPC (naive) or VL (KAPRA), to merge it:
        - with its nearest neighbour, that is, the group with the most similar label in `T_structure`;
        - with the group of size at least (2*`size` - |G|) that minimizes the metric.

    Parameters
    ----------
    :param T_postprocessed: list of dict of list of int
        List of good groups sanitized from `T_clustered`
    """

def postprocessing(algorithm, time_series_clustered=None, tree_structure=None, 
                                              partition_size=None, maximum_value=None, minimum_value=None,
                                              time_series_postprocessed=None):
    """
    Postprocessing to adjust group with size < partition_size

    :param algorithm: [description], defaults to "naive"
    :param time_series_clustered: [description], defaults to None
    :param tree_structure: [description], defaults to None
    :param partition_size: [description], defaults to None
    :param maximum_value: [description], defaults to None
    :param minimum_value: [description], defaults to None
    :param time_series_postprocessed: [description], defaults to None
    """
    
    index_change = list()
    group_change = list()
    tree_structure_change = list()

    for index_group_1, g_group_1 in enumerate(time_series_clustered):
        g_size = len(g_group_1)
        if g_size < partition_size:
            g_group_1_values = list(g_group_1.values())
            group_label = tree_structure[index_group_1]
            index_neighbour = -1
            measure_neighbour = float('inf') 
            for index_label, label in enumerate(tree_structure): # Nearest neighbour
                    if label[:-1] == group_label[:-1]: 
                        if index_label != index_group_1: 
                            if index_label not in index_change:
                                index_neighbour = index_label
            
            if index_neighbour > 0:
                table_1 = g_group_1_values + list(time_series_clustered[index_neighbour].values())
                
                if algorithm == "naive":
                    measure_neighbour = normalized_certainty_penalty(table=table_1, maximum_value=maximum_value, minimum_value=minimum_value)
                if algorithm == "kapra":
                    measure_neighbour = instant_value_loss(table=table_1)

                group_merge_neighbour = dict()
                group_merge_neighbour.update(g_group_1)
                group_merge_neighbour.update(time_series_clustered[index_neighbour])

            measure_other_group = float('inf')   

            for index, other_group in enumerate(time_series_clustered): 
                if len(other_group) >= 2*partition_size - g_size: #2k - |G|   
                    if index not in index_change:    
                        g_group_2 = g_group_1.copy()
                        for round in range(partition_size - g_size): #k - |G|         
                            round_measure = float('inf')
                            g_group_2_values = list(g_group_2.values())
                            for key, time_series in other_group.items(): 
                                if key not in g_group_2.keys(): 
                                
                                    if algorithm == "naive":
                                        temp_measure = normalized_certainty_penalty(table=g_group_2_values + [time_series], 
                                                                                            maximum_value=maximum_value, 
                                                                                            minimum_value=minimum_value)
                                    if algorithm == "kapra":
                                        temp_measure = instant_value_loss(table=g_group_2_values + [time_series])


                                    if temp_measure < round_measure:
                                        round_measure = temp_measure #set new min
                                        dict_to_add = { key : time_series }
                            
                            g_group_2.update(dict_to_add)

                        if round_measure < measure_other_group: # last ncp : ncp of the group
                            measure_other_group = round_measure 
                            group_merge_other_group = g_group_2
                            group_merge_remain = {key: value for (key, value) in other_group.items() if key not in g_group_2.keys()} 
                            index_other_group = index

            if measure_neighbour < measure_other_group: 
                index_change.append(index_neighbour)
                group_change.append(group_merge_neighbour)
                tree_structure_change.append(tree_structure[index_neighbour][:-1]) 

            else:
                index_change.append(index_other_group)
                group_change.append(group_merge_other_group)
                group_change.append(group_merge_remain)
                tree_structure_change.append("") # empty label


            index_change.append(index_group_1)
    
    time_series_clustered = [group for (index, group) in enumerate(time_series_clustered) if index not in index_change ]
    time_series_clustered += group_change 

    tree_structure = [label for (index, label) in enumerate(tree_structure) if index not in index_change]
    tree_structure += tree_structure_change

    bad_group_count = 0
    for index, group in enumerate(time_series_clustered):
        if len(group) < partition_size:
            bad_group_count +=1

    time_series_postprocessed += time_series_clustered
    
    if bad_group_count > 0:
        top_down_greedy_clustering_postprocessing(algorithm=algorithm, time_series_clustered=time_series_postprocessed, 
                                                  tree_structure=tree_structure, partition_size=partition_size, 
                                                  maximum_value=maximum_value, minimum_value=minimum_value)

def create_tree():
    # good leaf nodes
    good_leaf_nodes = list()
    bad_leaf_nodes = list()
    # creation root and start splitting node
    logger.info("Create-tree phase: initialization and start node splitting")
    node = Node(level=1, group=k_group, paa_value=paa_value)
    node.start_splitting(p_value, max_level, good_leaf_nodes, bad_leaf_nodes) 
    logger.info("Create-tree phase: finish node splitting")

    logger.info("Create-tree phase: start postprocessing")
    #for x in good_leaf_nodes:
    #   logger.info("Good leaf node {}, {}".format(x.size, x.pattern_representation))
    #for x in bad_leaf_nodes:
    #   logger.info("Bad leaf node {}".format(x.size))
    if len(bad_leaf_nodes) > 0:
    #    logger.info("Add bad node {} to good node, start postprocessing".format(len(bad_leaf_nodes)))
        Node.postprocessing(good_leaf_nodes, bad_leaf_nodes)
    #    for x in good_leaf_nodes:
    #        logger.info("Now Good leaf node {}, {}".format(x.size, x.pattern_representation))
    logger.info("Create-tree phase: finish postprocessing")
    for node in good_leaf_nodes:
        pr = node.pattern_representation
        for key in node.group:
            prs_dict[key] = pr


    # create-tree phase
    good_leaf_nodes = list()
    bad_leaf_nodes = list()

    # creation root and start splitting node
    logger.info("Create-tree phase: initialization and start node splitting with entire dataset")
    node = Node(level=1, group=time_series_dict, paa_value=paa_value)
    node.start_splitting(p_value, max_level, good_leaf_nodes, bad_leaf_nodes)
    logger.info("Create-tree phase: finish node splitting")

    # recycle bad-leaves phase
    logger.info("Start recycle bad-leaves phase")
    suppressed_nodes = list()
    if(len(bad_leaf_nodes) > 0):
        Node.recycle_bad_leaves(p_value, good_leaf_nodes, bad_leaf_nodes, suppressed_nodes, paa_value)
    logger.info("Finish recycle bad-leaves phase")
    suppressed_nodes_list = list()
    for node in suppressed_nodes:
        suppressed_nodes_list.append(node.group) # suppressed nodes!!!
    
    # group formation phase
    # preprocessing
    logger.info("Start group formation phase")
    pattern_representation_dict = dict() 
    p_group_list = list() 
    for node in good_leaf_nodes: 
        p_group_list.append(node.group)
        pr = node.pattern_representation
        for key in node.group:
            pattern_representation_dict[key] = pr