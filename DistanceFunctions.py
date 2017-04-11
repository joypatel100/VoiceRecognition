import numpy as np

'''
Returns pairwise distance matrix between points clouds pc0 and pc1.
'''
def pairwise_matrix(pc0, pc1, func):
    obs0 = len(pc0)
    obs1 = len(pc1)
    res = np.zeros([obs0, obs1])
    for i in range(obs0):
        for j in range(obs1):
            res[i][j] = dist_functions[func](pc0[i], pc1[j])
    return res

'''
Finds Euclidean distance between vectors u and v.
'''
def euclidean(u, v):
    return np.linalg.norm(u - v)

dist_functions = {
    "euclidean" : euclidean
}
