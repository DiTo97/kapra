import pandas as pd

from loguru import logger
from pathlib import Path

# Custom imports #
from .anonymized_dataset import AnonymizedDataset

DOWNSAMPLED_DIR = 'downsampled'
ANONYMIZED_DIR = 'anonymized'

def usage():
    print("[*] Usage: python k_P_anonymity.py <algorithm> <k_value>"
            + " <P_value> <paa_value> <l_value> <dataset>")
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

def generate_output_path(data_path):
    """
    Generate output path for anonymized dataset

    Parameters
    ----------
    data_path : string
        path of the original dataset

    Returns
    -------
    outpath : Windows path
        output path for anonymized dataset

    """
    
    abs_data_path = Path(data_path).absolute()

    # Compute output file path with '_anon' suffix
    outfilename = abs_data_path.parts[-1].replace('.csv', '_anon.csv')

    # Handle datasets coming from downsampled dir
    if abs_data_path.parent.parts[-1] == DOWNSAMPLED_DIR:
        parent_path = abs_data_path.parent.parent
    else:
        parent_path = abs_data_path.parent

    outdir = parent_path / ANONYMIZED_DIR
    outpath = (outdir / outfilename)

    return outpath 

def load_dataset(path: str, anonym=False):
    """
    Load original/anonymized dataset

    Parameters
    ----------
    path : str
        Dataset path.
    anonym : boolean, optional
        Set True if loading an anonymized dataset. The default is False.

    Returns
    -------
    QI_min_vals : TYPE
        DESCRIPTION.
    QI_max_vals : TYPE
        DESCRIPTION.
    QI_dict : dict
        dictionary containing time series
    A_s_dict : dict
        dictionary containing sensitive attributes for each time series
    col_names_QI : list
        column names of QI_dict time series

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

    # Remove sensitive attribute from original dataset only
    if anonym:
        
        # Extract sensitive data (A_s) DF
        logger.info('Extracted attribute ' + cols[-2] +
                ' as sensitive data')
        A_s_df = df.pop(cols[-2])
        cols.pop(-2)        
        
    else:
    
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
    
    col_names_QI = list(df.columns)

    return QI_min_vals, QI_max_vals, QI_dict, A_s_dict, col_names_QI

def save_anonymized_dataset(data_path, algorithm,
        prs = dict(), anonymized = list(), 
        sensitive = dict(), suppressed = list(), 
        col_names=list()):
    """
    Aggregate all separate k- and P- groups into a single anonymized dataset and save it to file.

    Parameters
    ----------
    :param data_path: str
        Relative path to the original dataset

    :param algorithm: str
        "naive" or "kapra", will be added to anonymized file name
        
    :param prs: dict of str - {}
        Dict of per-P-group SAX pattern representations

    :param anonymized: list of dict - []
        List of anonymized P-groups of records

    :param sensitive: dict of int - {}
        Dict of sensitive attributes (A_s)

    :param suppressed: list of dict - []
        List of P-groups of records to suppress (KAPRA-only)
    """

    outpath = generate_output_path(data_path)

    anonymized_dataset = AnonymizedDataset(anonymized,
            prs, suppressed, sensitive)

    anonymized_dataset.construct()
    anonymized_dataset.save(outpath, col_names)

    return outpath
