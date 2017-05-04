
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
from sklearn.neighbors import KNeighborsClassifier

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

def t_test_2(folder, file1, file2, start, end, dist_func=None, feature_name=None, df=None):
    class1 = extract(folder, file1, start, end)
    class2 = extract(folder, file2, start, end)
    if df == None:
        sample1 = record_self_distances(class1, dist_func, feature_name)
        sample2 = record_distances(class1, class2, dist_func, feature_name)
        print sample1
        print sample2
        print stats.ttest_ind(sample1,sample2, equal_var=False)
    else:
        for func,feature in df:
            print "Self Distances {0} {1}".format(func, feature)
            sample1 = record_self_distances(class1, func, feature)
            print "Distances {0} {1}".format(func, feature)
            sample2 = record_distances(class1, class2, func, feature)
            print "Testing Results {0} {1}".format(func, feature)
            #print sample1
            #print sample2
            print stats.ttest_ind(sample1,sample2, equal_var=False)


def machine_learning(folder, files, start, end, dist_func="cross_correlation", feature_name="raw_data", split=0.6, df=None):
    Y = []
    T = []
    train_end = int(start + split*(end-start))
    test_start = train_end + 1
    classes = []
    print "Extracting Training"
    for i in range(len(files)):
        print i
        infos = extract(folder, files[i], start, train_end)
        for info in infos:
            classes.append(info)
            Y.append(i)
    testing = []
    testing_correct = []
    print "Extracting Testing"
    res = []
    for i in range(len(files)):
        print i
        infos = extract(folder, files[i], test_start, end)
        for info in infos:
            testing.append(info)
            testing_correct.append(i)
    if df == None:
        print "Computing Training Distances"
        X = pairwise_matrix_symmetric(classes, classes, dist_func, feature_name)
        print "Computing Test Distances"
        T = pairwise_matrix(testing, classes, dist_func, feature_name)
        clf = SVC(kernel="precomputed")
        clf.fit(X, Y)
        print "predicting"
        pred = clf.predict(T)
        print pred
    else:
        correct = lambda pred, testing_correct : 1.0*sum([1.0 for i in range(len(pred)) if pred[i] == testing_correct[i]])/len(pred)
        for (func, feature) in df:
            print "Computing Training Distances {0} {1}".format(func, feature)
            X = pairwise_matrix_symmetric(classes, classes, func, feature)
            print "Computing Test Distances {0} {1}".format(func, feature)
            T = pairwise_matrix(testing, classes, func, feature)
            print "Correct Labels"
            print testing_correct

            print "Training SVM {0} {1}".format(func, feature)
            clf = SVC(kernel="precomputed")
            clf.fit(X, Y)
            print "Predicting SVM {0} {1}".format(func, feature)
            pred = clf.predict(T)
            print pred
            c = correct(pred, testing_correct)
            print c
            res.append(c)

            print "Training KNN {0} {1}".format(func, feature)
            clf = KNeighborsClassifier(n_neighbors=3, metric="precomputed")
            clf.fit(X, Y)
            print "Predicting KNN {0} {1}".format(func, feature)
            pred = clf.predict(T)
            print pred
            c = correct(pred, testing_correct)
            print c
            res.append(c)
    return res


if __name__ == '__main__':
    #t_test_2("Unlock","2017_03_30_au_{0}.wav","2017_03_30_ju_{0}.wav", 1, 5, "cross_correlation", "raw_data")
    #t_test_2("Unlock","2017_03_30_au_{0}.wav","2017_03_30_ju_{0}.wav", 1, 5, "multiscale_kernel", "pd_1d")
    #machine_learning("Unlock",["2017_03_30_au_{0}.wav","2017_03_30_ju_{0}.wav"],1,5, "cross_correlation", "raw_data")
    #machine_learning("Unlock",["2017_03_30_au_{0}.wav","2017_03_30_ju_{0}.wav"],1,5, "multiscale_kernel", "pd_1d")

    #t_test_2("Open Sesame","OS Loreanne/OS Loreanne {0}.wav","OS Eden/OS Eden {0}.wav", 1, 40, "cross_correlation", "raw_data")
    #t_test_2("Open Sesame","OS Loreanne/OS Loreanne {0}.wav","OS Eden/OS Eden {0}.wav", 1, 40, "multiscale_kernel", "pd_1d")
    #machine_learning("Open Sesame",["2017_03_30_og_{0}.wav","2017_03_30_jo_{0}.wav"],1,5, "cross_correlation", "raw_data")
    #machine_learning("Open Sesame",["2017_03_30_og_{0}.wav","2017_03_30_jo_{0}.wav"],1,5, "multiscale_kernel", "pd_1d")
    #machine_learning("Open Sesame",["OS Loreanne/OS Loreanne {0}.wav","OS Lijia/OS Lijia {0}.wav"],1,20, "cross_correlation", "raw_data")
    #f = ["OS Loreanne/OS Loreanne {0}.wav","OS Lijia/OS Lijia {0}.wav","OS Sam/OS Sam {0}.wav","OS Eden/OS Eden {0}.wav"]
    df = [("canberra","binned_pd_1d"),("braycurtis","binned_pd_1d"),("euclidean","binned_pd_1d"),("euclidean","pd_top_bars"),("inv_cross_correlation", "raw_data"), ("multiscale_kernel", "pd_1d")]
    t_test_2("Open Sesame","OS Sam/OS Sam {0}.wav","OS Eden/OS Eden {0}.wav", 1, 40,df=df)
    #print machine_learning("Open Sesame",f,1,40, df = df)
    '''
    res = []
    for i in range(len(f)):
        for j in range(i+1,len(f)):
            res.append(np.array(machine_learning("Open Sesame",[f[i],f[j]],1,40, df = df)))
            print res
    print res
    print np.mean(np.array(res),axis=0)
    '''
    #machine_learning("Open Sesame",["OS Loreanne/OS Loreanne {0}.wav","OS Lijia/OS Lijia {0}.wav"],1,40, df = [("canberra","binned_pd_1d"),("braycurtis","binned_pd_1d"),("euclidean","binned_pd_1d"),("euclidean","pd_top_bars"),("cross_correlation", "raw_data"),("inv_cross_correlation", "raw_data"), ("multiscale_kernel", "pd_1d")])
