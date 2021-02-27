"""
(k, P)-anonymity implementation of the KAPRA algorithm, from Shou et al. 2011,
Supporting Pattern-preserving Anonymization for Time-series Data, 5.3
"""

from loguru import logger

# Custom imports #
from .k_anonymity import k_anonymity_bottom_up
from .l_diversity import enforce_l_diversity
from .common import create_tree
from .io import load_dataset
from .io import save_anonymized_dataset
from .l_diversity import enforce_l_diversity
from .common import create_tree

def KAPRA(K_value, P_value, paa_value, l_value, data_path):
    """
    k-P anonymity based on work of Shou et al. 2013,
    Supporting Pattern-Preserving Anonymization for Time-Series Data

    Implementation of KAPRA approach

    Parameters
    ----------

    :param K_value: int
        K-requirement for (k, P) anonymity

    :param P_value: int
        P-requirement for (k, P) anonymity

    :param paa_value: int
        Number of real numbers used to encode the feature vector representation of each time series to be anonymized.
        For a reference of the encoding procedure, read the following paper:
        Lin, J., Keogh, E., Lonardi, S., & Chiu, B. (2003, June). A symbolic representation of time series, 
        with implications for streaming algorithms. 
        The above paper contains also a reference to the SAX pattern representation mentioned in Shou et al. 2013.

    :param l_value: int
        l-requirement for l-diversity (sensitive data perturbation)

    :param data_path: string
        Path of the dataset to be anonymized on disk
    """
    _, _, QI_time_series, A_s_dict, col_names = load_dataset(data_path)

    # create-tree phase
    logger.info("Start KAPRA create-tree phase ... ")

    PR = dict() # All pattern representations
                # from QI records

    P_subgroups, suppressed_groups = create_tree('kapra', QI_time_series, PR, P_value, paa_value)

    
    logger.info('End KAPRA create-tree phase')

    logger.info("Start group formation phase ... ")

    # List containing K-groups, each expressed as a dictionary of pairs (time series identifier, time series values)
    K_groups = list()

    # Call group formation algorithm 
    k_anonymity_bottom_up(P_subgroups, P_value, K_value, K_groups)

    enforce_l_diversity(PR, A_s_dict, K_groups, l_value)

    outpath = save_anonymized_dataset(data_path, "kapra", PR , K_groups, A_s_dict, col_names=col_names)

    logger.info('Saved anonymized dataset at: ' + str(outpath))
