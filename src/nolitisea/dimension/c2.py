import numpy as np
from scipy.spatial import cKDTree

from nolitisea.core.embed import delay_embedding


def correlation_integral(ts: np.ndarray, max_emb: int = 10, delay: int = 1, 
                         theiler: int = 0, eps: float = 0.01) -> dict:
    """
    Estimates the correlation sum for different embedding dimensions 
    at a single, fixed radius (epsilon).
    
    This implementation replaces the box-assisted search with a KD-Tree 
    spatial index for finding neighbors within the specified radius, utilizing 
    the Chebyshev distance (maximum norm).

    Parameters
    ----------
    ts : np.ndarray
        1-dimensional time series array.
    max_emb : int
        Maximum embedding dimension to test.
    delay : int
        Time delay between coordinates.
    theiler : int
        Theiler window to exclude temporally correlated neighbors.
    eps : float
        The fixed spatial radius (epsilon) threshold for the correlation sum.

    Returns
    -------
    dict
        A dictionary where keys are embedding dimensions and values are 
        the calculated correlation sum (c2) for the given epsilon.
        
    References
    ----------
    .. [1] Grassberger, P., & Procaccia, I. (1983). Characterization of strange 
           attractors. Physical review letters, 50(5), 346.
    .. [2] Hegger, R., Kantz, H., & Schreiber, T. (1999). Practical implementation 
           of nonlinear time series methods: The TISEAN package. Chaos: An 
           Interdisciplinary Journal of Nonlinear Science, 9(2), 413-435.
    """
    ts = np.ascontiguousarray(ts, dtype=np.float64)
    results = {}
    
    for m in range(1, max_emb + 1):
        emb_m = delay_embedding(ts, m, delay)
        n_points = len(emb_m)
        
        if n_points <= 0:
            break
            
        # Build the KD-Tree structure
        tree = cKDTree(emb_m)
        
        # count_neighbors with p=np.inf uses the Chebyshev distance.
        # Passing a scalar 'eps' returns a scalar count of all pairs (i, j) 
        # where distance <= eps, including self-pairs (i=j) and duplicates (j,i).
        raw_counts = tree.count_neighbors(tree, eps, p=np.inf)
        
        # Convert to strictly unique pairs (i < j)
        counts = (raw_counts - n_points) // 2
        
        # Total theoretically possible unique pairs
        valid_total = (n_points * (n_points - 1)) // 2
        
        # Apply Theiler window correction via vectorized array shifts
        if theiler > 0:
            for k in range(1, theiler + 1):
                if k >= n_points:
                    break
                    
                # Compute distances for pairs separated by exactly k time steps
                temporal_dists = np.max(np.abs(emb_m[:-k] - emb_m[k:]), axis=1)
                
                # Count how many of these excluded pairs fall under the radius threshold
                excluded_counts = np.sum(temporal_dists <= eps)
                
                # Subtract these pairs from the valid correlation sum counts
                counts -= excluded_counts
                valid_total -= len(temporal_dists)
                
        # Calculate the final probability distribution (correlation sum)
        c2 = counts / valid_total if valid_total > 0 else 0.0
        
        results[m] = c2
        
    return results