"""
(k, P)-anonymity implementation of the naive algorithm, from Shou et al. 2011,
Supporting Pattern-preserving Anonymization for Time-series Data, 5.2
"""

from loguru import logger

# Custom imports #
from .io import load_dataset, save_anonymized_dataset

def Naive(k_value, P_value, paa_value, data_path):
    QI_min_vals, QI_max_vals, QI_dict, A_s_dict = load_dataset(data_path)
    
    logger.info('Launching naive (k, P)-anonymity algorithm...')

    # start k_anonymity_top_down
    time_series_k_anonymized = list()
    tree_structure = list() # for postprocessing
    time_series_dict_copy = time_series_dict.copy()
    logger.info("Start k-anonymity top down approach")

    top_down_greedy_clustering(algorithm="naive", time_series=time_series_dict_copy, partition_size=k_value,
                                    maximum_value=attributes_maximum_value, minimum_value=attributes_minimum_value,
                                    time_series_clustered=time_series_k_anonymized, tree_structure=tree_structure)
    logger.info("End k-anonymity top down approach")

    logger.info("Start postprocessing k-anonymity top down approach")   
    time_series_postprocessed = list()
    top_down_greedy_clustering_postprocessing(algorithm="naive", time_series_clustered=time_series_k_anonymized, 
                                                tree_structure=tree_structure, partition_size=k_value, 
                                                maximum_value=attributes_maximum_value, minimum_value=attributes_minimum_value,
                                                time_series_postprocessed=time_series_postprocessed)
    logger.info("End postprocessing k-anonymity top down approach")

    time_series_k_anonymized = time_series_postprocessed

    # start kp anonymity
    pattern_representation_dict = dict()
    k_group_list = list()

    for group in time_series_k_anonymized:
        # append group to anonymized_data (after we will create a complete dataset anonymized)
        k_group_list.append(group) 
        # good leaf nodes
        good_leaf_nodes = list()
        bad_leaf_nodes = list()
        # creation root and start splitting node
        logger.info("Create-tree phase: initialization and start node splitting")
        node = Node(level=1, group=group, paa_value=paa_value)
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
                pattern_representation_dict[key] = pr

    save_anonymized_dataset(data_path, pattern_representation_dict,
            k_group_list, A_s_dict)
