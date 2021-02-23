from loguru import logger

# Custom imports #
from .common import top_down_greedy_clustering
from .common import top_down_greedy_clustering_postprocessing
from .common import postprocessing
from .common import find_group_with_min_vl

def k_anonymity_top_down(QI_dict, k, QI_k_anonymized,
        QI_max_vals, QI_min_vals):
    """
    Top down greedy k-anonymity implementation, from Xu et al. 2006,
    Utility-based Anonymization for Privacy Preservation with Less Information Loss, 4.2
    """

    if QI_max_vals == None or QI_min_vals == None:
        logger.error('No QI attribute boundaries are available, but they are required by the top down'
                + ' greedy k-anonymity algorithm to compute the NPC metric')
        exit(1)

    # 1. Top down greedy clustering
    QI_tree_structure = list()

    top_down_greedy_clustering('naive', QI_dict, k, QI_k_anonymized,
            QI_tree_structure, 'o', QI_max_vals, QI_min_vals)

    # 2. Postprocess bad leaves
    QI_postprocessed = list()
    
    postprocessing(algorithm='naive', time_series_clustered=QI_k_anonymized, 
                                                tree_structure=QI_tree_structure, partition_size=k, 
                                                maximum_value=QI_max_vals, minimum_value=QI_min_vals,
                                                time_series_postprocessed=QI_postprocessed)
    
    QI_k_anonymized = QI_postprocessed # Return to correct data structure

def k_anonymity_bottom_up(p_subgroups, p, k, tsid_pr_dict, gl):


    pgl = list() # PGL list described in the paper, implemented as a list of dictionaries, each having mappings
    # (time series identifier, pattern representation)

    # tsid_pr_dict: a dictionary with mapping (time series identifier, pattern representation) 
    for p_subgroup in p_subgroups: 
        # node.group: contains associations (time series identifier, time series values)
        pgl.append(p_subgroup.group)
        pr = p_subgroup.pattern_representation
        for ts_id in p_subgroup.group:
            tsid_pr_dict[ts_id] = pr

    splitted_p_subgroup = list()
    # List containing the indexes of the groups to be removed from the PGL list. Each element of PGL contains all the time
    # series associated with one good leaf node
    p_subgroups_splitted_idxs = list()

    # Loop over the time series of each p-subgroup, implements the preprocessing stage
    for p_subgroup_idx, p_subgroup in enumerate(pgl): 

        # if a p-subgroup can be splitted
        if len(p_subgroup) >= 2*p:
            
            tree_structure = list()
            p_group_splitted = list()
            # p_group_to_split is set to the current p-subgroup to be splitted
            p_group_to_split = p_subgroup

            # start top down greedy clustering
            # This partitioning process is targeted at minimizing the 
            # total instant value loss in the partitions. The resultant partitions, each regarded as a new P-subgroup, 
            # will be added into PGL to replace s_i. (see last two lines of code inside this for loop)
            top_down_greedy_clustering(algorithm="kapra", time_series=p_group_to_split, partition_size=p, 
                                    time_series_clustered=p_group_splitted, tree_structure=tree_structure)

            # p_group_splitted will contain multiple groups, splitted according to top_down_greedy_clustering and
            # generated from input group p_group

            time_series_postprocessed = list()
            # The top down greedy search method includes a post-processing phase, whose objective is to 
            # adjust the groups so that each group has at least k tuples; in this case, the partition size is chosen to
            # be p, which is the P requirement for (k, P) anonymity
            top_down_greedy_clustering_postprocessing(algorithm="kapra", time_series_clustered=p_group_splitted, 
                                                        tree_structure=tree_structure, partition_size=p,
                                                        time_series_postprocessed=time_series_postprocessed)
                                                            
            # Concatenate the list of all the postprocessed groups generated from p_group to list p_group_to_add
            splitted_p_subgroup += time_series_postprocessed
            p_subgroups_splitted_idxs.append(p_subgroup_idx) # add the index of the old group p_group to index_to_remove

    # remove from PGL all the groups whose indexes are included in p_subgroups_splitted_idxs
    pgl = [group for (index, group) in enumerate(pgl) if index not in p_subgroups_splitted_idxs]
    # add to the PGL list the p_group_to_add, after deleting the old group, which has been splitted!
    pgl += splitted_p_subgroup

    # gl: list GL from paper, which contains the k-groups
    # p_subgroups_k_promoted_idxs: indexes of the p-subgroups which are eligible to be promoted to k groups
    p_subgroups_k_promoted_idxs = list() 

    # step 1
    # loop over the p-groups in p_group_list
    for index, group in enumerate(pgl):
        # All P-subgroups in PGL containing no fewer than k time series are taken as k-groups and simply moved into GL (they are
        # deleted from PGL).
        # we recall that node.group is dictionary containing mappings (time series id, time series values)
        # len(group): number of time series inside a group
        if len(group) >= k:
            p_subgroups_k_promoted_idxs.append(index)
            gl.append(group)

    # delete newly found k-groups from PGL
    pgl = [group for (index, group) in enumerate(pgl) if index not in p_subgroups_k_promoted_idxs]

    # step 2 - 3 - 4
    p_subgroups_k_merged_idxs = list()
    # compute the length of all the p-subgroups left in PGL
    p_group_list_size = sum([len(group) for group in pgl])

    # paper while loop: while |PGL| >= k_value
    while p_group_list_size >= k:
        # find the P-subgroup s1 with the minimum instant value loss, and then create a new group G = s1.
        # k_group is group G in paper
        k_group, index_min = find_group_with_min_vl(group_to_search=pgl, 
                                                            index_ignored=p_subgroups_k_merged_idxs)
        # flag the previously found s_1 to be later removed
        p_subgroups_k_merged_idxs.append(index_min)
        # decrease the p-subgroup list size by the length of the previously found k-group
        # note that each subgroup used to generate the final G should be deleted from the subgroup list
        p_group_list_size -= len(k_group)

        # Find another P-subgroup which if merged with G, produces the minimal value loss of the union of the two group
        while len(k_group) < k:
            group_to_add, index_group_to_add = find_group_with_min_vl(group_to_search=pgl,
                                                                                group_to_merge=k_group, 
                                                                                index_ignored=p_subgroups_k_merged_idxs)
            # again, flag the corresponding group in PGL to be later removed
            p_subgroups_k_merged_idxs.append(index_group_to_add)
            # merge the time series of the two k-groups
            k_group.update(group_to_add) 
            # decrease the size of the PGL list
            p_group_list_size -= len(group_to_add)
        # put group G into list GL
        gl.append(k_group)   

    # step 5
    # remove all the p-subgroups which have been added to k-groups, by using the index list built before
    p_group_remaining = [group for (index, group) in enumerate(pgl) if index not in p_subgroups_k_merged_idxs]

    # for each remaining p-subgroup
    for p_group in p_group_remaining:
        # find the k-group which minimizes the instant value loss
        k_group, index_k_group = find_group_with_min_vl(group_to_search=gl,
                                                                group_to_merge=p_group)
        # remove the k-group from the list gl (the k-group list)
        gl.pop(index_k_group)
        # add the same k_group to the list again, this time with the time series of the added p-subgroup
        gl.update(p_group)
        gl.append(k_group)