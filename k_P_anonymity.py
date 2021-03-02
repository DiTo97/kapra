"""
(k, P)-anonymity implementations, from Shou et al. 2011,
Supporting Pattern-preserving Anonymization for Time-series Data
"""

import sys
import time
import os

import pandas as pd

from loguru import logger
from pathlib import Path

# Custom imports #
from includes.naive import Naive
from includes.kapra import KAPRA

from includes.io import usage

from includes.metric import global_anon_value_loss

from includes.pattern_loss import global_pattern_loss
from includes.pattern_loss import generate_output_path

RES_DIR = 'results'

if __name__ == "__main__":
    if not len(sys.argv) == 7:
        usage()

    # 1. Parse arguments
    algorithm = sys.argv[1].lower()

    k_value = int(sys.argv[2])
    P_value = int(sys.argv[3])
    paa_value = int(sys.argv[4])
    l_value = int(sys.argv[5])

    data_path = sys.argv[6]

    if k_value < P_value:
        logger.error('<k_value> must be greater or equal than <P_value>')
        usage()
    
    # 2. Execute (k, P) algorithm
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
    eta = round(float(end - start), 3) # Elapsed time

    # 3. Create results dir, if non-existent
    abs_root_path = Path(__file__).absolute().parent
    os.makedirs(abs_root_path / RES_DIR, exist_ok=True) 

    # 4. Compute pattern loss (PL)
    logger.info('Computing pattern loss...')

    global_ploss, global_ploss_avg = global_pattern_loss(data_path)

    tot_pattern_loss = round(float(global_ploss), 3)
    avg_pattern_loss = round(float(global_ploss_avg), 3)

    logger.info('Computed pattern loss of ' + str(avg_pattern_loss))

    # 5. Compute instant value loss (VL)
    logger.info('Computing instant value loss...')

    anonym_path = generate_output_path(data_path)
    glob_vl, mean_vl = global_anon_value_loss(anonym_path)

    tot_value_loss = round(float(glob_vl), 3)
    avg_value_loss = round(float(mean_vl), 3)

    logger.info('Computed instant value loss of ' + str(avg_value_loss))

    # 6. Save results as CSV file
    results_df = pd.DataFrame(columns = [ 'eta',
            'tot_pattern_loss', 'avg_pattern_loss',
            'tot_value_loss', 'avg_value_loss' ]) 
  
    results_df.loc[0] = [ eta,
            tot_pattern_loss, avg_pattern_loss,
            tot_value_loss, avg_value_loss ]

    abs_data_path = Path(data_path).absolute()

    outfilename = abs_data_path.parts[-1].replace('.csv', '') \
            + '_' + algorithm + '_k' + str(k_value)           \
            + '_P' + str(P_value) + '_paa' + str(paa_value)   \
            + '_l' + str(l_value) + '.csv'

    outfilepath = abs_root_path / RES_DIR / outfilename
    results_df.to_csv(outfilepath, sep =',', index=False) 

    print('\nFinalized (k, P) algorithm - ETA: ' + str(eta) + ' sec')
