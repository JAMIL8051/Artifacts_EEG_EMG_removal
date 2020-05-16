def p_empirical(data, n_clusters):
    "Empirical symbol distribution"
    #Arg: data: numpy array with size of length of microstate sequence
    # n_clusters: no. of ms clusters
    #returns: p: empirical distribution

    p = np.zeros(n_clusters)
    n = len(data)
    for i in range(n):
        p[data[i]] += 1.0
    p /= n
    return p


def T_empirical(data, n_clusters):
    """Empirical transition of the maps from one class to another.
    
    Args: data: numpy array of the length of ms sequence obtained
          n_clusters: no. of ms maps or class or clusters
    
    Returns: T: empirical transition matrix
        """
    T = np.zeros((n_clusters,n_clusters))
    n = len(data)
    for i in range(n-1):
        T[data[i],data[i+1]] += 1.0
    p_row = np.sum(T, axis = 1)
    for i in range(n_clusters):
        if (p_row[i] != 0.0):
            for j in range(n_clusters):
                T[i,j]/=p_row[i]
    return T


def print_matrix(T):
    """Console output of T matrix
    Args: T: matrix to print
    """
    for i,j in np.ndindex(T.shape):
        if(j==0):
            print("\t\t[{:.3f}".format(T[i,j]))
        elif (j == T.shape[1]-1):
            print("{:.3f}]\n".format(T[i,j]))
        else:
            print("{:.3f}".format(T[i,j]))
    return None