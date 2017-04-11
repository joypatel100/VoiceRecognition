
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


if __name__ == '__main__':
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
