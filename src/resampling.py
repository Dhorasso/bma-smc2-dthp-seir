import numpy as np
import warnings

def resampling_style(weights, name_method):
    """ Performs resampling algorithms used by particle filters based on the chosen method.

    Parameters
    ----------
    weights : list-like of float
        List of weights as floats.
    N : int
        Number of particles.
    name_method : str
        Name of the resampling method to be used. Should be one of: 'residual', 'stratified', 'systematic', 'multinomial'.

    Returns
    -------
    indexes : ndarray of ints
        Array of indexes into the weights defining the resample. i.e. the index of the zeroth resample is indexes[0], etc.

    Raises
    ------
    ValueError
        If the specified resampling method is not recognized.

    References
    ----------
        Copyright 2015 Roger R Labbe Jr.
        FilterPy library.
        http://github.com/rlabbe/filterpy
    
    """


    if name_method == 'residual':
        N = len(weights)
        indexes = np.zeros(N, 'i')

        # take int(N*w) copies of each weight, which ensures particles with the
        # same weight are drawn uniformly
        num_copies = (np.floor(N * np.asarray(weights))).astype(int)
        k = 0
        for i in range(N):
            for _ in range(num_copies[i]):  # make n copies
                indexes[k] = i
                k += 1
                if k>N:
                    warnings.warn("Resampling failed: Index k exceeds N. Adjusting indexes.")
                    return np.arange(N) # Reset j to 0 to avoid out-of-bounds error

        # use multinormal resample on the residual to fill up the rest. This
        # maximizes the variance of the samples
        residual = weights - num_copies  # get fractional part
        residual /= sum(residual)  # normalize
        cumulative_sum = np.cumsum(residual)
        cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
        indexes[k:N] = np.searchsorted(cumulative_sum, np.random.rand(N - k))

        return indexes

    elif name_method == 'stratified':
        N = len(weights)
        # make N subdivisions, and chose a random position within each one
        positions = (np.random.rand(N) + range(N)) / N

        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
                if j >= N:
                    warnings.warn("Resampling failed: Index j exceeds N. Adjusting indexes.")
                    return np.arange(N) # Reset j to 0 to avoid out-of-bounds error
        return indexes

    elif name_method == 'systematic':
        N = len(weights)

        # make N subdivisions, and choose positions with a consistent random offset
        positions = (np.random.rand() + np.arange(N)) / N

        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
                if j >= N:
                    warnings.warn("Resampling failed: Index j exceeds N. Adjusting indexes.")
                    return np.arange(N) # Reset j to 0 to avoid out-of-bounds error
        return indexes

    elif name_method == 'multinomial':
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
        return np.searchsorted(cumulative_sum, np.random.rand(len(weights)))

    else:
        raise ValueError("Unknown resampling method. Please choose one of: 'residual', 'stratified', 'systematic', 'multinomial'.")
