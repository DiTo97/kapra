import sys
import pandas as pd

from loguru import logger
from pathlib import Path

DOWNSAMPLED_DIR = 'downsampled'

# 1. Parse arguments
dataset_to_downsample = sys.argv[1]
records_to_keep = list(map(int, sys.argv[2].split(','))) # Cast to int an entire list

# 2. Read dataset as Pandas DF
path = Path(dataset_to_downsample)
df = pd.read_csv(path)

filename = path.parts[-1]
tot_num_records = len(df)

# 3. Extract and store downsampled datasets
for record in records_to_keep:
    if record > tot_num_records:
        logger.error('Cannot extract ' + str(record) + ' records from ' + filename
                + ': only ' + str(tot_num_records) + ' available')
        continue

    df_downsampled = df.head(record)

    filename_downsampled = filename.replace('.csv', '_' + str(record) + '.csv')
    path_downsampled = path.parent.absolute() / DOWNSAMPLED_DIR / filename_downsampled 

    df_downsampled.to_csv(path_downsampled, index=False)
