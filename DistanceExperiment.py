
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


hopsize = 1024
def e1(XMusic, FsMusic):
	(MFCC, MFCCDT, P) = getMFCC(XMusic, FsMusic, hopsize)
	return MFCCDT[3]

def p1(XMusic, FsMusic):
	return (20, ((FsMusic /2)/(float(hopsize)*20)), 1)

if __name__ == '__main__':
    '''
    folder = "Password"
    features = lambda folder, person, num : [getRips("{0}/2017_03_30_{1}_{2}.wav".format(folder, person, i)) for i in range(1,num+1)]
    person0 = features(folder,"ap",5)
    person1 = features(folder,"jp",5)
    print "Austin Top 30 Bars H0"
    print person0
    print "Joy Top 30 Bars H0"
    print person1
    d = pairwise_matrix(person0, person0 + person1, "euclidean")
    print "distance from Austin to Austin"
    print d[:,0:5]
    print "distance from Austin to Joy"
    print d[:,5:10]
    '''
    extractor = WavFeatureExtractor([e1], [p1], 0, 0, 0)
    folder = "Unlock"
    class1 = []
    class2 = []
    for i in range(1,6):
        c1 = extractor.extract_features_from_wav('{0}/2017_03_30_au_{1}.wav'.format(folder,i))
        c2 = extractor.extract_features_from_wav('{0}/2017_03_30_ju_{1}.wav'.format(folder,i))
        class1.append(c1[0])
        class2.append(c2[0])
    d1 = pairwise_matrix(class1, class1, "cross_correlation", "raw_data")
    d2 = pairwise_matrix(class1, class2, "cross_correlation", "raw_data")
    print d1
    print d2
