import numpy as np
from scipy.spatial import cKDTree  # type: ignore


def sample_entropy(time_series, m=2, r_multiplier=0.2):
    """
    Calculate the Sample Entropy (SampEn) utilizing cKDTree for O(N log N) speed.
    This is highly recommended for large datasets (N > 10,000).
    
    Reference:
    Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis 
    using approximate entropy and sample entropy. American Journal of 
    Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
    
    Parameters:
    time_series : array_like, 1D time series data.
    m : int, embedding dimension (default is 2).
    r_multiplier : float, coefficient for the threshold (default is 0.2).
    
    Returns:
    sampen : float, the calculated sample entropy value.
    """
    data = np.asarray(time_series, dtype=np.float64)
    N = len(data)
    
    # Calculate the tolerance threshold 'r' based on standard deviation
    std_data = np.std(data, ddof=1)
    # Fix: zero variance directly return inf (no valid fluctuation)
    if std_data < 1e-12:
        return 0.0
    r = r_multiplier * std_data
    
    # Construct template matrices for dimensions 'm' and 'm+1'
    # We treat each template as a point in m-dimensional and (m+1)-dimensional space
    templates_m = np.array([data[i : i + m] for i in range(N - m)])
    templates_m1 = np.array([data[i : i + m + 1] for i in range(N - m)])
    
    # Build KD-Trees for spatial indexing
    # leafsize is a tuning parameter; default is usually fine, but 30-50 can be optimal
    tree_m = cKDTree(templates_m)
    tree_m1 = cKDTree(templates_m1)
    
    # query pairs within distance 'r' using Chebyshev distance (p=np.inf)
    # The count_neighbors method efficiently counts pairs without storing them.
    # Note: count_neighbors returns ALL pairs (i, j) where dist <= r, including self-matches (i=j).
    count_m = tree_m.count_neighbors(tree_m, r=r, p=np.inf)
    count_m1 = tree_m1.count_neighbors(tree_m1, r=r, p=np.inf)
    
    # Subtract self-matches. 
    # There are exactly (N - m) self-matches because we compare the tree with itself.
    B = count_m - (N - m)
    A = count_m1 - (N - m)
    
    # Compute the final Sample Entropy
    if B <= 0 or A <= 0:
        return np.inf
        
    sampen = -np.log(A / B)
    return sampen