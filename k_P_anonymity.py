"""
(k, P)-anonymity implementations, from Shou et al. 2011,
Supporting Pattern-preserving Anonymization for Time-series Data
"""

import sys

from loguru import logger

# Custom imports #
from includes.naive import Naive
from includes.kapra import KAPRA

from includes.io import usage

import time

if __name__ == "__main__":
    if not len(sys.argv) == 7:
        usage()

    # Parse arguments
    algorithm = sys.argv[1].lower()

    k_value = int(sys.argv[2])
    P_value = int(sys.argv[3])
    paa_value = int(sys.argv[4])
    l_value = int(sys.argv[5])

    data_path = sys.argv[6]

    if k_value < P_value:
        logger.error('<k_value> must be greater or equal than <P_value>')
        usage()
    



    start = time.time()

    if algorithm == 'naive':
        Naive(k_value, P_value, paa_value, l_value, data_path)
    elif algorithm == 'kapra':
        KAPRA(k_value, P_value, paa_value, l_value, data_path)
    else:
        logger.error('Cannot interpret ' + algorithm
                + ' as a (k, P)-anonymity algorithm: only naive and KAPRA are supported')
        usage()

    end = time.time()

    print("Elapsed time in seconds : " + str(end - start) + "")
