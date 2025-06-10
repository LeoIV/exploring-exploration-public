import scipy
import numpy as np
cimport numpy as np

cpdef np.ndarray[np.float32_t, ndim=2] exploration_tsp(np.ndarray[np.float32_t, ndim=2] X):
    """
    Movement of observation center over time.
    Args:
        X (np.ndarray): shape (T, D)
    Returns:
        Z (np.ndarray): shape (T, 1)
    """
    #T, D = X.shape
    cdef int T = X.shape[0]
    cdef int D = X.shape[1]
    cdef np.ndarray[np.float32_t, ndim=1] tsp_solution = np.zeros(shape=(T,), dtype=np.float32)

    # Initialize with the first point's TSP solution
    cdef list current_path  = [0, 0]
    cdef float cumulative_distance = 0.0
    tsp_solution[0] = cumulative_distance

    cdef float best_distance_increase
    cdef float dist_increase
    cdef int best_insertion_index
    cdef np.ndarray[np.float32_t, ndim=2] dist_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X)).astype(np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] inner_dist_matrix

    # Calculate distances incrementally
    for t in range(1, T):
        # Update path by finding the best place to insert the new point
        best_distance_increase = float('inf')
        best_insertion_index = -1
        # Update distance matrix to include the new point
        inner_dist_matrix = dist_matrix[:t+1, :t+1]


        # Try inserting the new point in each position in the current path
        for i in range(len(current_path) - 1):
            # Calculate distance if new point were inserted between path[i] and path[i+1]
            dist_increase = (inner_dist_matrix[current_path[i], t] +
                             inner_dist_matrix[t, current_path[i+1]] -
                             inner_dist_matrix[current_path[i], current_path[i+1]])
            if dist_increase < best_distance_increase:
                best_distance_increase = dist_increase
                best_insertion_index = i + 1

        # Insert the new point at the best position found
        if best_insertion_index != -1:
            current_path.insert(best_insertion_index, t)
        cumulative_distance += best_distance_increase
        tsp_solution[t] = cumulative_distance

    return tsp_solution