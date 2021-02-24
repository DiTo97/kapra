"""
(k, P)-anonymity implementation of the naive algorithm, from Shou et al. 2011,
Supporting Pattern-preserving Anonymization for Time-series Data, 5.2
"""

from loguru import logger

# Custom imports #
from .k_anonymity import k_anonymity_top_down
from .l_diversity import 

from .common import create_tree

from .io import load_dataset
from .io import save_anonymized_dataset

def Naive(k_value, P_value, paa_value, data_path):
    QI_min_vals, QI_max_vals, QI_dict, A_s_dict = load_dataset(data_path)
    
    logger.info('Launching naive (k, P)-anonymity algorithm...')

    logger.info('Starting top down k-anonymity...')

    QI_k_anonymized = list() # All k-groups from QI records

    k_anonymity_top_down(QI_dict.copy(), k_value,     # Copy QI_dict because top down k-anonymity                                  
           QI_k_anonymized, QI_max_vals, QI_min_vals) # will delete its entries while forming groups

    logger.info('Ended top down k-anonymity')

    logger.info('Splitting P-subgroups from ' + len(QI_k_anonymized) + ' k-groups...')

    PR = dict() # All pattern representations
                # from QI records

    for idx, k_group in enumerate(QI_k_anonymized):
        logger.info('Create-tree phase k-group #' + str(idx) + '...')
        create_tree('naive', k_group, PR, P_value, paa_value)
        logger.info('Ended Create-tree k-group #' + str(idx))

    logger.info('Split all P-subgroups')

    # TODO: Enforce l-diversity
       

    save_anonymized_dataset(data_path, PR,
            QI_k_anonymized, A_s_dict)
