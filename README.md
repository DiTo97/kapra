﻿# KAPRA

Implementation of the [KAPRA](https://ieeexplore.ieee.org/abstract/document/6095556) algorithm for (k, P)-anonymity with l-diversity.

**Authors:** F. Minutoli, M. Ghirardelli, S. Bagnato, and G. Losapio.

```console
             ,--._,--.
           ,'  ,'   ,-`.
(`-.__    /  ,'   /
 `.   `--'        \__,--'-.
   `--/       ,-.  ______/
     (o-.     ,o- /
      `. ;        \
       |:          \
      ,'`       ,   \
     (o o ,  --'     :
      \--','.        ;
       `;;  :       /
        ;'  ;  ,' ,'
        ,','  :  '
        \ \   :
         `                                                            
```

## Dataset

All datasets are accessible in the *data* folder.

- [Weekly sales transactions](https://archive.ics.uci.edu/ml/datasets/sales_transactions_dataset_weekly) from UCI.

- [News popularity in social media platforms](https://archive.ics.uci.edu/ml/datasets/News+Popularity+in+Multiple+Social+Media+Platforms) from UCI.

## Folder structure

## Usage

```console
[*] Usage: python k_P_anonymity.py <algorithm> <k_value> <P_value> <paa_value> <l_value> <dataset>
```

### Parameters explanation

- `algorithm`, the (k, P)-anonymity implementation: naive or KAPRA;
- `k_value`, the k-anonymity constraint value;
- `P_value`, the P-anonymity constraint value on pattern sub-groups;
- `paa_value`, the piece-wise aggregate approximation (PAA) value to control the dimensionality of PRs;
- `l_value`, the l-diversity constraint value.
  
