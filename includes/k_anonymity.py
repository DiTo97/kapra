def k_anonymity_top_down(T, size, T_clustered,
        T_max_vals, T_min_vals):
    """
    Top down greedy k-anonymity implementation, from Xu et al. 2006,
    Utility-based Anonymization for Privacy Preservation with Less Information Loss, 4.2
    """
    
    QI_k_anonymized = list()
    QI_dict_copy = QI_dict.copy() # Copy dict because top down k-anonymity
                                  # will delete its entries while forming groups

    QI_tree_structure = list() # For postprocessing bad leaves

    logger.info('Starting top down k-anonymity...')

    top_down_greedy_clustering(algorithm='naive', time_series=QI_dict_copy,
            partition_size=k_value, maximum_value=QI_max_vals, minimum_value=QI_min_vals,
            time_series_clustered=QI_k_anonymized, tree_structure=QI_tree_structure)

    logger.info('Ended top down k-anonymity')

    logger.info('Starting top down k-anonymity postprocessing...')   
    QI_postprocessed = list()
    
    top_down_greedy_clustering_postprocessing(algorithm='naive', time_series_clustered=QI_k_anonymized, 
                                                tree_structure=QI_tree_structure, partition_size=k_value, 
                                                maximum_value=QI_max_vals, minimum_value=QI_min_vals,
                                                time_series_postprocessed=QI_postprocessed)
    logger.info("Ended top down k-anonymity postprocessing")

def k_anonymity_bottom_up():
    pass