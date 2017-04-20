import numpy as np
import math

def record_distances(pc0, pc1, func, feature_name):
    res = []
    for i in range(pc0):
        for j in range(pc1):
            res.append(dist_functions[func](pc0[i].features[feature_name], pc1[j].features[feature_name]))
    return res

def record_self_distances(pc0, func, feature_name):
    res = []
    for i in range(len(pc0)):
        for j in range(i+1, len(pc0)):
            res.append(dist_functions[func](pc0[i].features[feature_name], pc1[j].features[feature_name]))
    return res

'''
Returns pairwise distance matrix between points clouds pc0 and pc1.
'''
def pairwise_matrix(pc0, pc1, func, feature_name):
    obs0 = len(pc0)
    obs1 = len(pc1)
    res = np.zeros([obs0, obs1])
    for i in range(obs0):
        for j in range(obs1):
            res[i][j] = dist_functions[func](pc0[i].features[feature_name], pc1[j].features[feature_name])
    return res

'''
Finds Euclidean distance between vectors u and v.
'''
def euclidean(u, v):
    return np.linalg.norm(u - v)

'''
Finds the cross correlation between raw inputs u and v.
'''
def cross_correlation(u, v):
    return max(np.correlate(u,v))

'''
Multiscale kernel distance between 2 persistence diagrams P and Q
'''
def multiscale_kernel(P, Q):
    kernel = 0.0
    norm = lambda x, y : math.pow(math.pow(x[0]-y[0],2) + math.pow(x[1]-y[1],2),0.5)
    for p in P:
        for q in Q:
            kernel += math.pow(math.e,-1.0*math.pow(norm(p,q),2)/(8.0*sigma)) - math.pow(math.e,-1.0*math.pow(norm(p,np.array([q[1],q[0]])),2)/(8.0*sigma))
    return kernel/(8.0*math.pi*sigma)

dist_functions = {
    "euclidean" : euclidean,
    "cross_correlation" : cross_correlation,
    "multiscale_kernel" : multiscale_kernel
}
