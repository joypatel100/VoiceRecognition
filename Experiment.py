
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

hopsize = 50
dim = 20
def e1(XMusic, FsMusic):
    (MFCC, MFCCDT, P) = getMFCC(XMusic, FsMusic, hopsize)
    return MFCCDT[3]

def p1(XMusic, FsMusic):
	return (dim, ((FsMusic /2)/(float(hopsize)*dim)), 1)

def extract(folder, filename, start, end):
    extractor = WavFeatureExtractor([e1], [p1], 0, 0, 0)
    res = []
    for i in range(start, end+1):
        f = extractor.extract_features_from_wav('{0}/{1}'.format(folder,filename.format(i)))
        res.append(f[0])
    return res

def t_test_2(folder, file1, file2, start, end, dist_func, feature_name):
    class1 = extract(folder, file1, start, end)
    class2 = extract(folder, file2, start, end)
    sample1 = record_self_distances(class1, dist_func, feature_name)
    sample2 = record_distances(class1, class2, dist_func, feature_name)
    print sample1
    print sample2
    print stats.ttest_ind(sample1,sample2, equal_var=False)

def machine_learning(folder, files, start, end, dist_func, feature_name):
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
    X = pairwise_matrix(classes, classes, dist_func, feature_name)
    testing = []
    for i in range(len(files)):
        infos = extract(folder, files[i], test_start, end)
        for info in infos:
            testing.append(info)
    T = pairwise_matrix(testing, classes, dist_func, feature_name)
    clf = SVC(kernel="precomputed")
    clf.fit(X, Y)
    print clf.predict(T)

if __name__ == '__main__':
    #t_test_2("Unlock","2017_03_30_au_{0}.wav","2017_03_30_ju_{0}.wav", 1, 5, "cross_correlation", "raw_data")
    #t_test_2("Unlock","2017_03_30_au_{0}.wav","2017_03_30_ju_{0}.wav", 1, 5, "multiscale_kernel", "pd_1d")
    #machine_learning("Unlock",["2017_03_30_au_{0}.wav","2017_03_30_ju_{0}.wav"],1,5, "cross_correlation", "raw_data")
    #machine_learning("Unlock",["2017_03_30_au_{0}.wav","2017_03_30_ju_{0}.wav"],1,5, "multiscale_kernel", "pd_1d")

    #t_test_2("Open Sesame","2017_03_30_og_{0}.wav","2017_03_30_jo_{0}.wav", 1, 5, "cross_correlation", "raw_data")
    #t_test_2("Open Sesame","2017_03_30_og_{0}.wav","2017_03_30_jo_{0}.wav", 1, 5, "multiscale_kernel", "pd_1d")
    machine_learning("Open Sesame",["2017_03_30_og_{0}.wav","2017_03_30_jo_{0}.wav"],1,5, "cross_correlation", "raw_data")
    machine_learning("Open Sesame",["2017_03_30_og_{0}.wav","2017_03_30_jo_{0}.wav"],1,5, "multiscale_kernel", "pd_1d")
