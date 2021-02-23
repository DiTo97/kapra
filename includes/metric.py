import numpy as np

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

def instant_value_loss(T):
    """
    Compute the instant value loss, VL(T), from Shou et al. 2011,
    Supporting Pattern-preserving Anonymization for Time-series Data, 4.2.2
    """ 

    r_plus  = list()
    r_minus = list()

    n = len(T[0]) # # of QI attributes in T

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
