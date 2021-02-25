from loguru import logger

# Custom imports #
from .common import top_down_greedy_clustering
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
    
    postprocessing('naive', k, QI_k_anonymized,
            QI_tree_structure, QI_postprocessed, QI_max_vals, QI_min_vals) 
    
    QI_k_anonymized = QI_postprocessed # Return to correct data structure

def k_anonymity_bottom_up(p_subgroups, p, k, GL):

    """
    Bottom up group formation procedure, from Shou et al. 2013,
    Supporting Pattern-Preserving Anonymization for Time-Series Data, 5.3.3

    Parameters
    ----------
    :param p_subgroups: List of dicts
        Each dict contained in list p_subgroups is formed by pairs (ts_id, ts_values)

    :param p: int
        P-requirement for (k, P) anonymity

    :param k: int
        K-requirement for (k, P) anonymity

    :param GL: List of dicts
        Resulting list of K-groups produced by k_anonymity_bottom_up. Filled after executing this procedure.
    """

    PGL = list() # PGL list described in the paper, implemented as a list of dictionaries, each having mappings
    # (time series identifier, pattern representation). Each dictionary represents a group.

    # List containing all the resulting subgroups produced by splitting a subgroup
    splitted_p_subgroups = list()

    # List containing the indexes of the groups to be removed from the PGL list after splitting. 
    p_subgroups_splitted_idxs = list()

    for p_subgroup in p_subgroups: 
        PGL.append(p_subgroup)

    # Loop over the time series of each p-subgroup. Implements the preprocessing stage.
    # Each group contains associations (time series identifier, time series values)
    for p_subgroup_idx, p_subgroup in enumerate(PGL): 

        # if a p-subgroup can be splitted
        if len(p_subgroup) >= 2*p:
            
            # Tree structure which needs to be filled by top_down_greedy_clustering, in order to later call the postprocessing
            # function on the results.
            postprocessing_clustering_tree = list()

            temp_splitted_p_subgroup = list()

            # p_subgroup_to_be_splitted is set to the current p-subgroup, which will be splitted because of its size (>=2*p)
            p_subgroup_to_be_splitted = p_subgroup

            # Start top down greedy clustering (as reported in the paper): split the current group in subgroups having size p
            top_down_greedy_clustering("kapra", p_subgroup_to_be_splitted, p, temp_splitted_p_subgroup, postprocessing_clustering_tree)

            # Initialize list containing postprocessed subgroups
            postprocessed_p_subgroups = list()

            # The top down greedy search method includes a post-processing phase, whose objective is to 
            # adjust the groups so that each group has at least k tuples; in this case, the partition size is chosen to
            # be p, which is the P requirement for (k, P) anonymity
            postprocessing('naive', p, temp_splitted_p_subgroup, postprocessing_clustering_tree,postprocessed_p_subgroups)
                                                            
            # Concatenate the list of all the postprocessed groups generated from the current p_subgroup to list splitted_p_subgroup
            # Splitted_p_subgroup will contain multiple groups, splitted according to top_down_greedy_clustering and postprocessed by
            # the postprocessing function
            splitted_p_subgroups += postprocessed_p_subgroups
            p_subgroups_splitted_idxs.append(p_subgroup_idx) # add the index of the old group p_subgroup to index_to_remove

    # remove from PGL all the p-subgroups whose indexes are included in p_subgroups_splitted_idxs
    # we recall that PGL contains dictionaries formed by associations (time series identifier, time series values)
    PGL = [p_subgroup for (p_subgroup_idx, p_subgroup) in enumerate(PGL) if p_subgroup_idx not in p_subgroups_splitted_idxs]

    # add to the PGL list newly formed subgroups contained in splitted_p_subgroup, 
    # after deleting the corresponding old group, which has been splitted!
    PGL += splitted_p_subgroups

    # gl: list GL from paper, which contains the k-groups
    # p_subgroups_k_promoted_idxs: indexes of the p-subgroups which are eligible to be promoted to be k-groups
    p_subgroups_k_promoted_idxs = list() 

    # loop over the p-subgroups in pgl
    for p_subgroup_idx, p_subgroup in enumerate(PGL):
        # All P-subgroups in PGL containing no fewer than k time series are taken as k-groups and simply moved into GL (they are
        # deleted from PGL).
        # we recall that node.group is a dictionary containing mappings (time series id, time series values)
        # len(group): number of time series inside a group
        if len(p_subgroup) >= k:
            p_subgroups_k_promoted_idxs.append(p_subgroup_idx)
            GL.append(p_subgroup)

    # Delete newly found k-groups from PGL (which contains the "old" p_subgroups)
    PGL = [p_subgroup for (p_subgroup_idx, p_subgroup) in enumerate(PGL) if p_subgroup_idx not in p_subgroups_k_promoted_idxs]

    p_subgroups_k_merged_idxs = list()
    # compute the length of all the p-subgroups left in PGL
    card_PGL = sum([len(p_subgroup) for p_subgroup in PGL])

    # paper while loop: while |PGL| >= k_value
    while card_PGL >= k:
        # find the P-subgroup s1 with the minimum instant value loss, and then create a new group G = s1.
        # Group G in paper and corresponding index (a k-group)
        G, G_idx = find_group_with_min_vl(group_to_search=PGL, 
                                                            index_ignored=p_subgroups_k_merged_idxs)
        # flag the previously found s_1 to be later removed from PGL list
        p_subgroups_k_merged_idxs.append(G_idx)
        # decrease the p-subgroup list size by the length of the previously found k-group G
        # note that each subgroup used to generate the final G should be deleted from the subgroup list after the completion
        # of the merging operation
        card_PGL -= len(G)

        while len(G) < k:
            # Find another P-subgroup which if merged with G, produces the minimal value loss of the union of the two groups
            S_min, S_min_idx = find_group_with_min_vl(PGL,G,p_subgroups_k_merged_idxs)
            # again, flag the corresponding group in PGL to be later removed
            p_subgroups_k_merged_idxs.append(S_min_idx)
            # merge the time series of the two k-groups
            G.update(S_min) 
            # decrease the size of the PGL list
            card_PGL -= len(S_min)
        # put group G into list GL
        GL.append(G) 

    # remove all the p-subgroups which have been added to k-groups, by using the index list built before
    p_subgroups_left = [p_subgroup for (p_subgroup_idx, p_subgroup) in enumerate(PGL) if p_subgroup_idx not in p_subgroups_k_merged_idxs]

    # for each remaining p-subgroup
    for p_subgroup in p_subgroups_left:
        # from paper: Each remaining P-subgroup in PGL will choose to join a k-group which again 
        # minimizes the total instant value loss
        G_prime, G_prime_idx = find_group_with_min_vl(GL,p_subgroup)
        # remove the k-group G_prime from list GL (the k-group list)
        GL.pop(G_prime_idx)
        # add the same k_group G_prime to the list again, this time with the time series of the added p-subgroup
        GL.update(p_subgroup)
        GL.append(G_prime)