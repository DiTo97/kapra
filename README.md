# KAPRA

Implementation of the [KAPRA](https://ieeexplore.ieee.org/abstract/document/6095556) algorithm for (k, P)-anonymity.

**Authors:** F. Minutoli, M. Ghirardelli, S. Bagnato, and G. Losapio.

## Tentative schedule

- Test naive (k, P)-anonymity implementation's correctness.
  
- Implement KAPRA algorithm
    1. P-subgroups clustering first;
    2. K-groups clustering with a bottom-up approach.
   
- Choose 3 meaningful real-life datasets from [Google DS](https://datasetsearch.research.google.com/).

- Test both (k, P) algorithms for utility and performance.
- Test both (k, P) algorithms for varying values of k and P.

## Presentation

- Briefly describe both (k, P) algorithms.

- Explain both (k, P) implementation details in depth.

- Discuss test results for utility and performance.
- Discuss test results for varying values of k and P.

## Dataset
Weekly sales transactions [dataset](https://archive.ics.uci.edu/ml/datasets/sales_transactions_dataset_weekly) from UCI, in the *data* folder.

## Usage 
```
[*] usage: python kp-anonymity.py k_value p_value paa_value data\sales_transactions_dataset_weekly.csv
```

### Parameters explanation
- *k_value*, the K-anonymity constraint value;
- *p_value*, the P-anonymity constraint value on pattern sub-groups.
- *paa_value*, the piece-wise aggregate approximation (PAA) value to reduce the dimensionality of patterns.