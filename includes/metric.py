import numpy as np
import pandas as pd 
from loguru import logger

def normalized_certainty_penalty(T, T_max_vals, T_min_vals):
    """
    Compute the normalized certainty penalty, NCP(T), from Xu et al. 2006,
    Utility-based Anonymization for Privacy Preservation with Less Information Loss, 3.2.1
    """

    z = list()
    y = list()
    A = list()

    n = len(T[0]) # # of QI attributes in T

    # 1. Loop over all QI attributes
    # to fill z, y, and A
    for i in range(n):
        A.append(abs(T_max_vals[i] - T_min_vals[i]))

        z_i = 0
        y_i = float('inf')
        
        # Find the best z_i and y_i values
        # across all records in T
        for row in T:
            if row[i] >= z_i:
                z_i = row[i]

            if row[i] < y_i:
                y_i = row[i]

        z.append(z_i) 
        y.append(y_i)

    # 2. Compute NCP(t) and then NCP(T)
    ncp_t = 0

    for i in range(n):
        if A[i] == 0:
            ncp_t += 0
        else:
            ncp_t += (z[i] - y[i]) / A[i]

    ncp_T = len(T)*ncp_t 
    return ncp_T

def instant_value_loss(T, r_plus=None, r_minus=None):
    """
    Compute the instant value loss, VL(T), from Shou et al. 2011,
    Supporting Pattern-preserving Anonymization for Time-series Data, 4.2.2
    """ 

    n = len(T[0])  # # of QI attributes in T

    if not r_plus or not r_minus:
        r_plus  = list()
        r_minus = list()


        for i in range(n): 
            r_plus_i  = 0
            r_minus_i = float('inf')

            for row in T:
                if row[i] > r_plus_i:
                    r_plus_i = row[i]

                if row[i] < r_minus_i:
                    r_minus_i = row[i]

            r_plus.append(r_plus_i) 
            r_minus.append(r_minus_i)
    
    # Compute VL(t) and then VL(T)
    vl_t = 0

    for i in range(n):
        vl_t += pow((r_plus[i] - r_minus[i]), 2) / n

    vl_T = len(T)*np.sqrt(vl_t)
    return vl_T

def global_anon_value_loss(anonym_path):
    """given the nae of an anonymized dataset, loads it and computes
    instant value loss for whole table"""
    glob_vl = 0

    # loaf dataframe, group by last column
    df = pd.read_csv(anonym_path)
    g_df = df.groupby(["group"])
    for key, _ in g_df:
        # g is a dataframe with all the anon rows belonging to group key
        g = g_df.get_group(key)

        # remove Ids
        cols = list(g.columns)
        g.pop(cols[0])
        cols = cols[1:]

        # remove sensitive data, sax and group (last three columns)
        for _ in range(3):
            g.pop(cols[-1])
            cols.pop(-1)

        QI_list = [list(g.iloc[i]) for i in range(len(g))]  # Quasi-identifier attributes

        fst_row = list(g.iloc[0])

        r_plus  = [0 for _ in range(len(fst_row))]
        r_minus = [0 for _ in range(len(fst_row))]
        
        for i in range(len(fst_row)):  # max and min will be the same for all this anon envelope
            # remove "[" and "]"
            rng = fst_row[i]
            # get min and max
            mn, mx = rng[1:-1].split("|")
            r_minus[i] = int(mn)
            r_plus[i] = int(mx)

        glob_vl += instant_value_loss(QI_list, r_plus=r_plus, r_minus=r_minus)
    return glob_vl/len(df)
