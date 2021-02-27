"""
(k, P)-anonymity implementation of the naive algorithm, from Shou et al. 2011,
Supporting Pattern-preserving Anonymization for Time-series Data, 5.2
"""

from loguru import logger

# Custom imports #
from .k_anonymity import k_anonymity_top_down
from .l_diversity import enforce_l_diversity

from .common import create_tree

from .io import load_dataset
from .io import save_anonymized_dataset

def Naive(k_value, P_value, paa_value, l_value, data_path):
    QI_min_vals, QI_max_vals, QI_time_series, A_s_dict, col_names = load_dataset(data_path)
    
    # If k greater than the available QI data
    if k_value > len(A_s_dict):
        logger.error('<k_value> cannot be greater than the'
                + ' available QI time series data')
        exit(1)

    logger.info('Launching naive (k, P)-anonymity algorithm...')

    # 1. Create k-groups from whole QI data
    logger.info('Starting top down k-anonymity...')

    QI_k_anonymized = list() # All k-groups from QI records

    k_anonymity_top_down(QI_time_series.copy(), k_value, # Copy QI_time_series because top down k-anonymity                                  
           QI_k_anonymized, QI_max_vals, QI_min_vals)    # will delete its entries while forming groups

    logger.info('Ended top down k-anonymity')

    # 2. Create P-groups for each k-group
    logger.info('Splitting P-subgroups from ' + str(len(QI_k_anonymized)) + ' k-groups...')

    PR = dict() # All pattern representations
                # from QI records

    for idx, k_group in enumerate(QI_k_anonymized):
        logger.info('Create-tree phase k-group #' + str(idx) + '...')
        create_tree('naive', k_group, PR, P_value, paa_value)
        logger.info('Ended Create-tree k-group #' + str(idx))

    logger.info('Split all P-subgroups')

    # 3. Enforce l-diversity
    logger.info('Enforcing l-diversity...')

    enforce_l_diversity(PR, A_s_dict, QI_k_anonymized, l_value)

    logger.info('Enforced l-diversity')

    outpath = save_anonymized_dataset(data_path, "naive", PR, QI_k_anonymized, A_s_dict, col_names=col_names)

    logger.info('Saved anonymized dataset at: ' + str(outpath))
