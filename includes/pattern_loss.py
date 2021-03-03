# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 18:26:06 2021

@author: Gianvito
"""

import numpy as np
from loguru import logger
from scipy.spatial.distance import cosine

from .io import load_dataset, generate_output_path
from saxpy.paa import paa
from saxpy.znorm import znorm 
from saxpy.alphabet import cuts_for_asize
from saxpy.strfunc import idx2letter
from saxpy.sax import ts_to_string, sax_by_chunking

def letter2idx(letter):    
    return ord(letter) - 97


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


def global_pattern_loss(data_path, algorithm):
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
    anonym_path = generate_output_path(data_path, algorithm)
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
    

    
    
    
    
    
    
    