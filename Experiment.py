
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.interpolate as interp

from TDA import *
from SlidingWindow import *
from MusicFeatures import *
import scipy.io.wavfile
from MusicFeatures import *
from DistanceFunctions import *
from MusicSpeech import *
from WavFeatureExtractor import *
from scipy import stats
from sklearn.svm import SVC

def e1(XMusic, FsMusic):
    hopsize = 1024
	(MFCC, MFCCDT, P) = getMFCC(XMusic, FsMusic, hopsize)
	return MFCCDT[3]

def p1(XMusic, FsMusic):
	return (20, ((FsMusic /2)/(float(hopsize)*20)), 1)

def extract(folder, filename, start, end):
    extractor = WavFeatureExtractor([e1], [p1], 0, 0, 0)
    res = []
    for i in range(start, end+1):
        f = extractor.extract_features_from_wav('{0}/{1}'.format(folder,filename.format(i)))
        res.append(f[0])
    return res

def t_test_2(folder, file1, file2, start, end):
    class1 = extract(folder, file1, start, end)
    class2 = extract(folder, file2, start, end)
    sample1 = record_self_distances(class1, "euclidean", "raw_feature_data")
    sample2 = record_distances(class1, class2, "euclidean", "raw_feature_data")
    print stats.ttest_ind(sample1,sample2, equal_var=False)

def machine_learning(folder, files, start, end):
    Y = []
    T = []
    train_end = int(start + 0.8*(end-start))
    test_start = train_end + 1
    classes = []
    for i in range(len(files)):
        infos = extract(folder, files[i], start, train_end)
        for info in infos:
            classes.append(info)
            Y.append(i)
    X = pairwise_matrix(classes, classes, "multiscale_kernel", "pd_1d")
    testing = []
    for i in range(len(files)):
        infos = extract(folder, files[i], test_start, end)
        for info in infos:
            testing.append(info)
    T = pairwise_matrix(testing, classes, "multiscale_kernel", "pd_1d")
    clf = SVC()
    clf.fit(X, Y, kernel="precomputed")
    print clf.predict(testing)
