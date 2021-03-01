"""
(k, P)-anonymity implementations, from Shou et al. 2011,
Supporting Pattern-preserving Anonymization for Time-series Data
"""

from includes.pattern_loss import global_pattern_loss
from includes.metric import global_anon_value_loss
from includes.pattern_loss import generate_output_path
import sys

from loguru import logger

# Custom imports #
from includes.naive import Naive
from includes.kapra import KAPRA

from includes.io import usage

import time
import os

import pandas as pd

from pathlib import Path

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

    os.makedirs("results", exist_ok=True) 

    global_ploss, global_ploss_avg = global_pattern_loss(data_path)

    anonym_path = generate_output_path(data_path)

    glob_vl, mean_vl = global_anon_value_loss(anonym_path)

    print("Elapsed time in seconds : " + str(end - start) + "")

    results = pd.DataFrame(columns =['ElapsedTime', 'GlobalPatternLoss', 'AveragePatternLoss', 'GlobalValueLoss', 'MeanValueLoss']) 
  
  
    results.loc[0]=[float(end-start), float(global_ploss), float(global_ploss_avg), \
        float(glob_vl), float(mean_vl)]

    abs_data_path = Path(data_path).absolute()

    outfilename = abs_data_path.parts[-1].replace('.csv', '') + "_k" + str(k_value) + "_p" \
        + str(P_value) + "_paa" + str(paa_value) + "_l" + str(l_value) + ".csv"

    outfilepath = "./results/" + outfilename

    results.to_csv(outfilepath, sep ='\t') 
