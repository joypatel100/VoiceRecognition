import numpy as np

dist_functions = {
    "euclidean" : euclidean
}

'''
Returns pairwise distance matrix between points clouds pc0 and pc1.
'''
def pairwise_matrix(pc0, pc1, func):
    row0, col0 = pc0.shape
    row1, col1 = pc1.shape
    res = np.zeros(row0, row1)
    for i in range(row0):
        for j in range(row1):
            res[i][j] = dist_functions[func](pc0[i], pc1[j])
    return res

'''
Finds Euclidean distance between vectors u and v.
'''
def euclidean(u, v):
    return np.linalg.norm(u - v)
