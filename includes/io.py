import numpy as np
import pandas as pd

from loguru import logger
from pathlib import Path

# Custom imports #
from .anonymized_dataset import AnonymizedDataset

ANONYMIZED_DIR = 'anonymized'

def usage():
    print("[*] Usage: python k_P_anonymity.py <algorithm> <k_value>"
            + " <P_value> <paa_value> <dataset>")
    exit(1)

def get_min_max_QI_values_from_table(df, QI_cols):
    """
    Extract min and max values for each QI attribute in the table.
    
    Parameters
    ----------
    :param df: pd.DataFrame
        Time series data DF

    :param QI_cols: list of str
        QI attributes-only column names

    Returns
    -------
    :return QI_min_values: list of int
        List of min values for each QI attribute

    :return QI_max_values: list of int
        List of max values for each QI attribute
    """

    QI_max_values = list()
    QI_min_values = list()

    for col in QI_cols:
        QI_max_values.append(df[col].max())
        QI_min_values.append(df[col].min())

    return QI_min_values, QI_max_values

def load_dataset(path: str):
    """
    Parameters
    ----------
    :param path: str
        Relative path to the dataset
    """

    data_path = Path(path)

    if not data_path.is_file():
        logger.error(str(path.absolute())
                + ' not found')
        exit(1)

    logger.info('Loading dataset...')

    df = pd.read_csv(data_path) # Time series data DF

    # Extract column names
    cols = list(df.columns)[1:] # Leave column 0 out, as it contains Ids

    # Extract sensitive data (A_s) DF
    logger.info('Extracted attribute ' + cols[-1] +
            ' as sensitive data')
    A_s_df = df.pop(cols[-1])
    cols.pop(-1)

    # Convert DFs to dicts
    QI_dict = dict()  # Quasi-identifier attributes
    A_s_dict = dict() # Sensitive data

    for i, row in df.iterrows():
        A_s_dict[row[0]] = A_s_df[i]
        QI_dict[row[0]]  = list(row[cols])
  
    QI_min_vals, QI_max_vals = get_min_max_QI_values_from_table(df, cols)

    logger.info('Loaded dataset')

    return QI_min_vals, QI_max_vals, QI_dict, A_s_dict

def save_anonymized_dataset(data_path, prs = dict(),
        anonymized = list(), sensitive = dict(),
        suppressed = list()):
    """
    Aggregate all separate k- and P- groups into a single anonymized dataset and save it to file.

    Parameters
    ----------
    :param data_path: str
        Relative path to the original dataset

    :param prs: dict of str - {}
        Dict of per-P-group SAX pattern representations

    :param anonymized: list of dict - []
        List of anonymized P-groups of records

    :param sensitive: dict of int - {}
        Dict of sensitive attributes (A_s)

    :param suppressed: list of dict - []
        List of P-groups of records to suppress (KAPRA-only)
    """

    abs_data_path = Path(data_path).absolute()

    # Compute output file path with '_anon' suffix
    outfilename = abs_data_path.parts[-1].replace('.csv', '_anon.csv')
    outdir = abs_data_path.parent / ANONYMIZED_DIR

    outpath = outdir / outfilename

    anonymized_dataset = AnonymizedDataset(anonymized,
            prs, suppressed, sensitive)

    anonymized_dataset.construct()
    anonymized_dataset.save(outpath)
