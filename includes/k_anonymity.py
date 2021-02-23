from loguru import logger

# Custom imports #
from .common import top_down_greedy_clustering
from .common import postprocessing

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
                                                tree_structure=QI_tree_structure, partition_size=k_value, 
                                                maximum_value=QI_max_vals, minimum_value=QI_min_vals,
                                                time_series_postprocessed=QI_postprocessed)
    
    QI_k_anonymized = QI_postprocessed # Return to correct data structure

def k_anonymity_bottom_up():
    p_group_to_add = list()
    index_to_remove = list()

    for index, p_group in enumerate(p_group_list): 
        if len(p_group) >= 2*p_value:
            
            tree_structure = list()
            p_group_splitted = list()
            p_group_to_split = p_group 

            # start top down greedy clustering
            top_down_greedy_clustering(algorithm="kapra", time_series=p_group_to_split, partition_size=p_value, 
                                    time_series_clustered=p_group_splitted, tree_structure=tree_structure)

            #logger.info("Start postprocessing k-anonymity top down approach")
            time_series_postprocessed = list()
            top_down_greedy_clustering_postprocessing(algorithm="kapra", time_series_clustered=p_group_splitted, 
                                                        tree_structure=tree_structure, partition_size=p_value,
                                                        time_series_postprocessed=time_series_postprocessed)
                                                
            #logger.info("End postprocessing k-anonymity top down approach")
            
            p_group_to_add += time_series_postprocessed
            index_to_remove.append(index)
    
    
    p_group_list = [group for (index, group) in enumerate(p_group_list) if index not in index_to_remove ]
    p_group_list += p_group_to_add
    
    
    k_group_list = list()
    index_to_remove = list() 
    
    # step 1
    for index, group in enumerate(p_group_list):
        if len(group) >= k_value:
            index_to_remove.append(index)
            k_group_list.append(group)
    
    p_group_list = [group for (index, group) in enumerate(p_group_list) if index not in index_to_remove ]

    # step 2 - 3 - 4
    index_to_remove = list()
    p_group_list_size = sum([len(group) for group in p_group_list])
    
    while p_group_list_size >= k_value:
        k_group, index_min = find_group_with_min_value_loss(group_to_search=p_group_list, 
                                                            index_ignored=index_to_remove)
        index_to_remove.append(index_min)
        p_group_list_size -= len(k_group)

        while len(k_group) < k_value:
            group_to_add, index_group_to_add = find_group_with_min_value_loss(group_to_search=p_group_list,
                                                                                group_to_merge=k_group, 
                                                                                index_ignored=index_to_remove)
            index_to_remove.append(index_group_to_add)
            k_group.update(group_to_add) 
            p_group_list_size -= len(group_to_add)
        k_group_list.append(k_group)   
    
    # step 5
    p_group_remaining = [group for (index, group) in enumerate(p_group_list) if index not in index_to_remove ]
    
    for p_group in p_group_remaining:
        k_group, index_k_group = find_group_with_min_value_loss(group_to_search=k_group_list,
                                                                group_to_merge=p_group)
        k_group_list.pop(index_k_group)
        k_group.update(p_group)
        k_group_list.append(k_group)