import numpy as np
import math

def record_distances(pc0, pc1, func, feature_name):
    res = []
    for i in range(len(pc0)):
        for j in range(len(pc1)):
            res.append(dist_functions[func](pc0[i].features[feature_name], pc1[j].features[feature_name]))
    return res

def record_self_distances(pc0, func, feature_name):
    res = []
    for i in range(len(pc0)):
        for j in range(i+1, len(pc0)):
            res.append(dist_functions[func](pc0[i].features[feature_name], pc0[j].features[feature_name]))
    return res

'''
Returns pairwise distance matrix between points clouds pc0 and pc1.
'''
def pairwise_matrix_symmetric(pc0, pc1, func, feature_name):
    obs0 = len(pc0)
    obs1 = len(pc1)
    res = np.zeros([obs0, obs1])
    for i in range(obs0):
        for j in range(i+1,obs1):
            res[i][j] = dist_functions[func](pc0[i].features[feature_name], pc1[j].features[feature_name])
            res[j][i] = res[i][j]
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

def euclidean_pairwise_squared(X,Y):
	return np.sum(X**2,1)[:,None] + np.sum(Y**2,1)[None,:] - 2.0*np.dot(X,Y.T)

'''
Multiscale kernel distance between 2 persistence diagrams P and Q
'''
def multiscale_kernel(P,Q,sigma=1.0):
	Q_bar = np.array([Q[:,1],Q[:,0]]).T
	return np.sum(np.exp(-1.0*euclidean_pairwise_squared(P,Q)/(8.0*sigma)) - np.exp(-1.0*euclidean_pairwise_squared(P,Q_bar)/(8.0*sigma)))/(8.0*math.pi*sigma)

def multiscale_kernel_slow(P, Q, sigma=1.0):
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
