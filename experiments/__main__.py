import os
import subprocess

from concurrent.futures import ProcessPoolExecutor # Parallelize comparisons
from pathlib import Path

k_P_pairs = [
    (16, 6),
    (16, 3)
]

k_P_pairs = [
    (16, 9),
    (64, 6),
    (128, 6),
    (16, 12)
]

l   = 1 # No perturbation from l-diversity
PAA = 6

DATASETS = [
    'facebook_microsoft.csv',
    'facebook_palestine.csv',
    'sales_transactions_dataset_weekly.csv'
]

ALGORITHMS = [
    'naive',
    'kapra'
]

DATA_DIR = 'data'
SRC = 'k_P_anonymity.py'

abs_data_dir = Path(os.path.dirname(os.path.abspath('__file__'))).parent / DATA_DIR
abs_src_path = abs_data_dir.parent / SRC

def run_experiment(dataset):
    abs_data_path = str(abs_data_dir / dataset)
    errs = []

    for algo in ALGORITHMS:
        for k_P in k_P_pairs:
            k, P = k_P

            cmd = 'python {} {} {} {} {} {} {}'.format('"' + str(abs_src_path) + '"',
                    algo, k, P, PAA, l, '"' + abs_data_path + '"')

            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                errs.append(str(e))

    if len(errs) == 0:
        print('No errors found with dataset {}'.format(dataset))

    return errs

def print_experiment_errs(res):
    exper_errs = list(res)

    for exper in exper_errs:
        dataset, errs = exper

        if len(errs) == 0:
            continue

        print('Found {} errors with dataset {}'.format(len(errs), dataset))

        for err in errs:
            print(err)

        print('\n')

    return exper_errs

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=16) as pool:
        res = zip(DATASETS, list(pool.map(run_experiment, DATASETS))) # Store experiment results

    # Print experiment errors
    exper_errs = print_experiment_errs(res)
