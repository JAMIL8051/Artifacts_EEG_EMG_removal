def zero_mean(data, axis=0):
    """

    :type data: d-by-n matrices, where d is the number of features and n is the number of instances.
    """
    mean = data.mean(axis=1, keepdims=True)
    return data - mean