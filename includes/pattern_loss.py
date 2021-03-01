# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 18:26:06 2021

@author: Gianvito
"""

import numpy as np
from loguru import logger
from scipy.spatial.distance import cosine

from .io import load_dataset, generate_output_path


def znorm(series, znorm_threshold=0.01):
    """Znorm implementation.
    >>> print ['{:0.2f}'.format(x) for x in znorm([1, 2, 3])]
    ['-1.22', '0.00', '1.22']
    >>> print ['{:0.2f}'.format(x) for x in znorm([3, 2, 1])]
    ['1.22', '0.00', '-1.22']
    >>> print ['{:0.2f}'.format(x) for x in znorm([1, 2])]
    ['-1.00', '1.00']
    >>> print ['{:0.2f}'.format(x) for x in np.sum(znorm([[1, 2, 3], [6, 5, 4]]), axis=0)]
    ['0.00', '0.00', '0.00']
    >>> znorm([[1, 2, 3], [6, 5, 4]])
    array([[-1., -1., -1.],
           [ 1.,  1.,  1.]])
    """

    series = np.array(series)
    original_series_shape = series.shape
    is_multidimensional = (len(series.shape) > 1) and (series.shape[1] > 1)
    mu = np.average(series, axis=0)
    C = np.cov(series, bias=True, rowvar=not is_multidimensional)

    # Only update those subsequences with variance over the threshold.
    if is_multidimensional:
        series = series - mu
        C = np.diagonal(C)
        indexes = (C >= np.square(znorm_threshold))
        series[:, indexes] = (series[:, indexes] / np.sqrt(C[indexes]))
    else:
        series = series - mu
        if C >= np.square(znorm_threshold):
            series /= np.sqrt(C)

    # Check on shape returned.
    assert(series.shape == original_series_shape)

    return series



def paa(series, paa_segment_size, sax_type='unidim'):
    """PAA implementation.
    >>> paa([1, 2, 3], 3, 'unidim')
    array([1., 2., 3.])
    >>> paa([1, 2, 3], 1, 'unidim')
    array([2.])
    >>> paa([4, 3, 8, 5], 1, 'unidim')
    array([5.])
    >>> paa([[1, 2, 3], [6, 5, 4]], 1, 'repeat')
    array([[3.5, 3.5, 3.5]])
    >>> paa([[1, 2, 3], [6, 5, 4]], 2, 'repeat')
    array([[1., 2., 3.],
           [6., 5., 4.]])
    """

    series = np.array(series)
    series_len = series.shape[0]

    if sax_type in ['repeat', 'energy']:
        num_dims = series.shape[1]
    else:
        num_dims = 1
        is_multidimensional = (len(series.shape) > 1) and (series.shape[1] > 1)
        if not is_multidimensional:
            series = series.reshape(series.shape[0], 1)

    res = np.zeros((num_dims, paa_segment_size))

    for dim in range(num_dims):
        # Check if we can evenly divide the series.
        if series_len % paa_segment_size == 0:
            inc = series_len // paa_segment_size

            for i in range(0, series_len):
                idx = i // inc
                np.add.at(res[dim], idx, np.mean(series[i][dim]))
            res[dim] /= inc
        # Process otherwise.
        else:
            for i in range(0, paa_segment_size * series_len):
                idx = i // series_len
                pos = i // paa_segment_size
                np.add.at(res[dim], idx, np.mean(series[pos][dim]))
            res[dim] /= series_len

    if sax_type in ['repeat', 'energy']:
        return res.T
    else:
        return res.flatten()
    
    
    


def cuts_for_asize(a_size):
    """Generate a set of alphabet cuts for its size."""
    """ Typically, we generate cuts in R as follows:
        get_cuts_for_num <- function(num) {
        cuts = c(-Inf)
        for (i in 1:(num-1)) {
            cuts = c(cuts, qnorm(i * 1/num))
            }
            cuts
        }
        get_cuts_for_num(3) """
    options = {
        2: np.array([-np.inf,  0.00]),
        3: np.array([-np.inf, -0.4307273, 0.4307273]),
        4: np.array([-np.inf, -0.6744898, 0, 0.6744898]),
        5: np.array([-np.inf, -0.841621233572914, -0.2533471031358,
                    0.2533471031358, 0.841621233572914]),
        6: np.array([-np.inf, -0.967421566101701, -0.430727299295457, 0,
                    0.430727299295457, 0.967421566101701]),
        7: np.array([-np.inf, -1.06757052387814, -0.565948821932863,
                    -0.180012369792705, 0.180012369792705, 0.565948821932863,
                    1.06757052387814]),
        8: np.array([-np.inf, -1.15034938037601, -0.674489750196082,
                    -0.318639363964375, 0, 0.318639363964375,
                    0.674489750196082, 1.15034938037601]),
        9: np.array([-np.inf, -1.22064034884735, -0.764709673786387,
                    -0.430727299295457, -0.139710298881862, 0.139710298881862,
                    0.430727299295457, 0.764709673786387, 1.22064034884735]),
        10: np.array([-np.inf, -1.2815515655446, -0.841621233572914,
                     -0.524400512708041, -0.2533471031358, 0, 0.2533471031358,
                     0.524400512708041, 0.841621233572914, 1.2815515655446]),
        11: np.array([-np.inf, -1.33517773611894, -0.908457868537385,
                     -0.604585346583237, -0.348755695517045,
                     -0.114185294321428, 0.114185294321428, 0.348755695517045,
                     0.604585346583237, 0.908457868537385, 1.33517773611894]),
        12: np.array([-np.inf, -1.38299412710064, -0.967421566101701,
                     -0.674489750196082, -0.430727299295457,
                     -0.210428394247925, 0, 0.210428394247925,
                     0.430727299295457, 0.674489750196082, 0.967421566101701,
                     1.38299412710064]),
        13: np.array([-np.inf, -1.42607687227285, -1.0200762327862,
                     -0.736315917376129, -0.502402223373355,
                     -0.293381232121193, -0.0965586152896391,
                     0.0965586152896394, 0.293381232121194, 0.502402223373355,
                     0.73631591737613, 1.0200762327862, 1.42607687227285]),
        14: np.array([-np.inf, -1.46523379268552, -1.06757052387814,
                     -0.791638607743375, -0.565948821932863, -0.36610635680057,
                     -0.180012369792705, 0, 0.180012369792705,
                     0.36610635680057, 0.565948821932863, 0.791638607743375,
                     1.06757052387814, 1.46523379268552]),
        15: np.array([-np.inf, -1.50108594604402, -1.11077161663679,
                     -0.841621233572914, -0.622925723210088,
                     -0.430727299295457, -0.2533471031358, -0.0836517339071291,
                     0.0836517339071291, 0.2533471031358, 0.430727299295457,
                     0.622925723210088, 0.841621233572914, 1.11077161663679,
                     1.50108594604402]),
        16: np.array([-np.inf, -1.53412054435255, -1.15034938037601,
                     -0.887146559018876, -0.674489750196082,
                     -0.488776411114669, -0.318639363964375,
                     -0.157310684610171, 0, 0.157310684610171,
                     0.318639363964375, 0.488776411114669, 0.674489750196082,
                     0.887146559018876, 1.15034938037601, 1.53412054435255]),
        17: np.array([-np.inf, -1.5647264713618, -1.18683143275582,
                     -0.928899491647271, -0.721522283982343,
                     -0.541395085129088, -0.377391943828554,
                     -0.223007830940367, -0.0737912738082727,
                     0.0737912738082727, 0.223007830940367, 0.377391943828554,
                     0.541395085129088, 0.721522283982343, 0.928899491647271,
                     1.18683143275582, 1.5647264713618]),
        18: np.array([-np.inf, -1.59321881802305, -1.22064034884735,
                     -0.967421566101701, -0.764709673786387,
                     -0.589455797849779, -0.430727299295457,
                     -0.282216147062508, -0.139710298881862, 0,
                     0.139710298881862, 0.282216147062508, 0.430727299295457,
                     0.589455797849779, 0.764709673786387, 0.967421566101701,
                     1.22064034884735, 1.59321881802305]),
        19: np.array([-np.inf, -1.61985625863827, -1.25211952026522,
                     -1.00314796766253, -0.8045963803603, -0.633640000779701,
                     -0.47950565333095, -0.336038140371823, -0.199201324789267,
                     -0.0660118123758407, 0.0660118123758406,
                     0.199201324789267, 0.336038140371823, 0.47950565333095,
                     0.633640000779701, 0.8045963803603, 1.00314796766253,
                     1.25211952026522, 1.61985625863827]),
        20: np.array([-np.inf, -1.64485362695147, -1.2815515655446,
                     -1.03643338949379, -0.841621233572914, -0.674489750196082,
                     -0.524400512708041, -0.385320466407568, -0.2533471031358,
                     -0.125661346855074, 0, 0.125661346855074, 0.2533471031358,
                     0.385320466407568, 0.524400512708041, 0.674489750196082,
                     0.841621233572914, 1.03643338949379, 1.2815515655446,
                     1.64485362695147]),
    }

    return options[a_size]


   

"""Convert a normlized timeseries to SAX symbols."""


def idx2letter(idx):
    """Convert a numerical index to a char."""
    if 0 <= idx < 20:
        return chr(97 + idx)
    else:
        raise ValueError('A wrong idx value supplied.')
        

def letter2idx(letter):    
    return ord(letter) - 97
        
        
    
def ts_to_string(series, cuts):
    """A straightforward num-to-string conversion.
    >>> ts_to_string([-1, 0, 1], cuts_for_asize(3))
    'abc'
    >>> ts_to_string([1, -1, 1], cuts_for_asize(3))
    'cac'
    """

    series = np.array(series)
    a_size = len(cuts)
    sax = list()

    for i in range(series.shape[0]):
        num = series[i]

        # If the number is below 0, start from the bottom, otherwise from the top
        if num >= 0:
            j = a_size - 1
            while j > 0 and cuts[j] >= num:

                j = j - 1
            sax.append(idx2letter(j))
        else:
            j = 1
            while j < a_size and cuts[j] <= num:
                j = j + 1
            sax.append(idx2letter(j-1))

    return ''.join(sax)



def compute_fv(series, paa_size, znorm_threshold=0.01):
    """
    Compute feature vector (PAA representation)

    Parameters
    ----------
    series : list
        time series values
    paa_size : int
        Target size of the PAA representation
    znorm_threshold : float, optional
        threshold used for z-score normalization. The default is 0.01.

    Returns
    -------
    fv : np.ndarray
        feature vector (paa representation)

    """
    
    series_norm = znorm(series, znorm_threshold)
    fv = paa(series_norm, paa_size)
    
    return fv


def sax_by_chunking(series, paa_size, alphabet_size=3, znorm_threshold=0.01):
    """
    Compute SAX: Simple chunking conversion implementation.

    Parameters
    ----------
    series : TYPE
        DESCRIPTION.
    paa_size : TYPE
        DESCRIPTION.
    alphabet_size : TYPE, optional
        DESCRIPTION. The default is 3.
    znorm_threshold : TYPE, optional
        DESCRIPTION. The default is 0.01.

    Returns
    -------
    sax : TYPE
        DESCRIPTION.

    """
    
    # paa representation
    fv = compute_fv(series, paa_size, znorm_threshold)
    
    # to string
    cuts = cuts_for_asize(alphabet_size)
    sax = ts_to_string(fv, cuts)
    
    return sax




def empirical_median(paa_idx, seed=23, size=1000000):
    """
    Get the empirical median for each interval denoted by a breakpoint
    index

    Parameters
    ----------
    paa_idx : np.ndarray
        Array of (reconstructed) breakpoint indexes
    seed : int, optional
        Seed for gaussian random generation. The default is 23.
    size : int, optional
        Number of samples used to approximate the probabilistic medians
        of the intervals. The default is 1000000.

    Returns
    -------
    paa_reco : np.ndarray
        Array of empirical medians

    """
    
    # Init
    paa_reco = np.zeros(paa_idx.shape)
    
    # Estimate number of levels based on the given string
    level = np.max(paa_idx) + 1
    
    # if the string is composed by only a, skip the reconstruction and
    # simply return the zero vector
    if level > 1:
        
        # Get corresponding breakpoints
        breakpoints = cuts_for_asize(level)
        
        # Empirical probabilistic median
        np.random.seed(seed)
        pts = np.random.normal(size=size)
        
        
        # Get interval endpoints [beta_lo; beta_up)
        for i in range(len(paa_idx)):
            
            # due to how breakpoints are stored: [beta_0, ... , beta_{l-1}]
            start_idx = paa_idx[i]
            beta_lo = breakpoints[start_idx]
            if start_idx < level-1:
                beta_up = breakpoints[paa_idx[i]+1]
            else:
                beta_up = np.inf
            
            paa_reco[i] = np.median(pts[(pts >= beta_lo) & (pts < beta_up)])
    
    return paa_reco




def reconstruct_fv(pr):
    """
    Reconstruct the feature vector (PAA) given a pattern representation (SAX)

    Parameters
    ----------
    pr : string
        original pattern representation (SAX)
        
    Returns
    -------
    paa_reco : np.ndarray
        Reconstructed feature vector (PAA)

    """
    
    # Get breakpoint indexes
    paa_idx = np.array([letter2idx(x) for x in pr])
    
    # Reconstruct
    paa_reco = empirical_median(paa_idx)
    
    return paa_reco


def cosine_distance(u, v):
    """
    Scipy implementation
    0 -> the angle is 0 = closest
    1 -> the angle is 180 = furthest

    Parameters
    ----------
    u : np.ndarray
        array 1D
    v : np.ndarray
        array 1D

    Returns
    -------
    cd : float
        1 - cos(theta)

    """
    
    # If both vectors are different from zero compute cosine distance
    if (np.sum(u) > 0) & (np.sum(v) > 0):
        cd = cosine(u,v)
    
    # If both vectors are zero vectors the distance is 0 
    elif (np.sum(u) == 0) & (np.sum(v) == 0):
        cd = 0.
        
    # If one of the two vectors is 0 but the other one is not return 1
    else:
        cd = 1.
        
    return cd


def pattern_loss(series, pr, paa_size, znorm_threshold=0.01):
    """
    Pattern loss 

    Parameters
    ----------
    series : list
        Time-series
    pr : TYPE
        pattern representation (SAX string)
    paa_size : int
        size of the word w

    Returns
    -------
    pl : float
        pattern loss

    """
    
    # compute paa from original time-series
    p = compute_fv(series, paa_size, znorm_threshold)
    
    # reconstruct paa from SAX
    p_star = reconstruct_fv(pr)
    
    # Compute pattern loss   
    pl = cosine_distance(p, p_star)
    
    
    return pl, p, p_star


def global_pattern_loss(data_path):
    """
    Compute global pattern loss on a given dataset and its anonymized version

    Parameters
    ----------
    data_path : string
        path of the original dataset
    avg : boolean, optional
        Set true if you want an average loss instead of a simple sum.
        The default is False.

    Returns
    -------
    global_ploss : float
        Global pattern loss

    """
    
    # Load original time QI attributes
    _, _, QI_ts, _, _ = load_dataset(data_path)
    
    # Infer path of the anonymized dataset
    anonym_path = generate_output_path(data_path)
    if not anonym_path.is_file():
        logger.error(str(anonym_path.absolute())
                + ' not found')
        exit(1)
      
    # Load QI attributes of the anonymized dataset
    _, _, QI_ts_anonym, _, _ = load_dataset(anonym_path, anonym=True)
    
    
    # Compute pattern loss for each time series
    num_series = len(QI_ts)
    plosses = np.zeros((num_series,))
    
    for idx, k in enumerate(QI_ts.keys()):
        
        if k in QI_ts_anonym.keys():
            
            series = QI_ts[k]
            pr = QI_ts_anonym[k][-2] # sax
            paa_size = len(pr)
            
            pl, _, _  = pattern_loss(series,pr,paa_size)
            
            plosses[idx] = pl
            
        else:
            logger.info('Key {} missing'.format(k))
            
    global_ploss = np.sum(plosses)
    
    global_ploss_avg = global_ploss / num_series
    
    return global_ploss, global_ploss_avg
    

    
    
    
    
    
    
    