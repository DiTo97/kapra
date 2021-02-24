"""
(k, P)-anonymity implementation of the KAPRA algorithm, from Shou et al. 2011,
Supporting Pattern-preserving Anonymization for Time-series Data, 5.3
"""

from loguru import logger

# Custom imports #
from .io import load_dataset

def KAPRA(k_value, P_value, paa_value, l_value, data_path):
    _, _, QI_dict, A_s_dict = load_dataset(data_path)

    logger.info('Launching KAPRA (k, P)-anonymity algorithm...')
