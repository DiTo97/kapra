"""
(k, P)-anonymity implementation of the naive algorithm, from Shou et al. 2011,
Supporting Pattern-preserving Anonymization for Time-series Data, 5.2
"""

from loguru import logger

# Custom imports #
from .k_anonymity import k_anonymity_top_down

from .io import load_dataset
from .io import save_anonymized_dataset

def Naive(k_value, P_value, paa_value, data_path):
    QI_min_vals, QI_max_vals, QI_dict, A_s_dict = load_dataset(data_path)
    
    logger.info('Launching naive (k, P)-anonymity algorithm...')

    logger.info('Starting top down k-anonymity...')

    QI_k_anonymized = list()

    k_anonymity_top_down(QI_dict.copy(), k_value,     # Copy QI_dict because top down k-anonymity                                  
           QI_k_anonymized, QI_max_vals, QI_min_vals) # will delete its entries while forming groups

    logger.info('Ended top down k-anonymity')

    # start kp anonymity
    prs_dict = dict()
    k_groups_list = list()

    for k_group in QI_postprocessed:
        # append group to anonymized_data (after we will create a complete dataset anonymized)
        k_groups_list.append(k_group) 
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

    # TODO: Enforce l-diversity
       

    save_anonymized_dataset(data_path, prs_dict,
            k_groups_list, A_s_dict)
