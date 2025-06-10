import math

import numpy as np
cimport numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

cdef float knn_entropy(np.ndarray[np.float32_t, ndim=2] X, int k=3):
    """
    Estimate the Shannon entropy of a dataset using the k-ncpdef earest neighbors method.


    Parameters:
        X (numpy.ndarray): The N x D array of data points.
        k (int): Number of neighbors to use in the k-NN estimation.
        
    Returns:
        float: The estimated entropy.
    """
    cdef int N = X.shape[0]
    cdef int D = X.shape[1]
    cdef float entropy_estimate, avg_log_dist
    cdef np.ndarray[np.float32_t, ndim=2] distances
    cdef np.ndarray[np.float32_t, ndim=1] nn_distances

    tree = cKDTree(X)
    distances = tree.query(X, [k])[0].astype(np.float32)
    nn_distances = distances[:, -1]  # k-th nearest neighbor distance
    avg_log_dist = np.mean(np.log(nn_distances + 1e-10))  # Add small value to avoid log(0)
    # volume of the D-dimensional hypersphere
    V = np.pi ** (D / 2) / math.gamma(D / 2 + 1)

    # Calculate the entropy
    entropy_estimate = (
            digamma(N) - digamma(k) + np.log(V) + D * avg_log_dist
    )
    return entropy_estimate

cpdef exploration_entropy(np.ndarray[np.float32_t, ndim=2] X):
    """
    Calculate the empirical Shannon entropy over cumulative observation points at each time step,
    dynamically setting k based on the sample size.
    
    Parameters:
        X (numpy.ndarray): The T x D array where each row is a data point in [0, 1]^D.
    
    Returns:
        numpy.ndarray: An array of entropy values for each time step.
    """
    cdef int T = X.shape[0]
    cdef int D = X.shape[1]
    # Due to singularity, the first D points are ignored
    cdef np.ndarray[np.float32_t, ndim=1] entropies = np.zeros(T - D, dtype=np.float32)
    cdef int k, t

    for eidx, t in enumerate(range(D, T)):
        # Dynamically set k as the square root of current sample size
        k = max(1, int(np.log(t + 1)))

        # Estimate entropy using k-NN on cumulative data
        entropies[eidx] = knn_entropy(X[:t], k=k)

    return entropies
