"""
Programmer: Chris Tralie (ctralie@alumni.princeton.edu)
Purpose: To show how TDA can be used to quantify how periodic
an audio clip is.  Simple example with music versus speech.
Show how doing a delay embedding on raw audio is a bad idea when
the length of the period is on the order of seconds, and how
"audio novelty functions" come in handy
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.interpolate as interp

from TDA import *
from SlidingWindow import *
from MusicFeatures import *
import scipy.io.wavfile

def getRips(filename):
    FsMusic, XMusic = scipy.io.wavfile.read(filename)
    hopSize = 50
    novFnMusic = getAudioNovelty(XMusic, FsMusic, hopSize)
    dim = 20
    Tau = (FsMusic/2)/(float(hopSize)*dim)
    dT = 1
    Y = getSlidingWindowInteger(novFnMusic, dim, Tau, dT)
    Y = Y - np.mean(Y, 1)[:, None]
    Y = Y/np.sqrt(np.sum(Y**2, 1))[:, None]

    PDs = doRipsFiltration(Y, 1)
    d0 = PDs[0]
    d1 = PDs[1]
    #return d0[:,1] - d0[:,0], d1[:,1] - d1[:,0]
    delta = np.array(sorted(d0[:,1] - d0[:,0], reverse=True))[0:30]
    return delta

def binPD (PD, birth_max, death_max, birth_min=0.0, death_min=0.0, axis_bins=3):

    feature_vector = [0] * (axis_bins * axis_bins)
    for point in PD:
        birth_bin = int((point[0] - birth_min - 0.0001) / (float(birth_max - birth_min) / axis_bins))
        death_bin = int((point[1] - death_min - 0.0001) / (float(death_max - death_min) / axis_bins))
        print (point, birth_bin, death_bin)

        feature_vector[(birth_bin * axis_bins) + death_bin] += 1

    return feature_vector

if __name__ == '__main__':
    #Don't Stop Believing
    FsMusic, XMusic = scipy.io.wavfile.read("Austin/2017_03_30_ao_2.wav")
    FsSpeech, XSpeech = scipy.io.wavfile.read("2017_03_30_jp.wav")

    #Step 1: Try a raw delay embedding
    #Note that dim*Tau here spans a half a second of audio,
    #since Fs is the sample rate
    #dim = round(FsMusic/200)
    #Tau = 100
    #dT = FsMusic/125
    #Y = getSlidingWindowInteger(XMusic[0:FsMusic*3], dim, Tau, dT)
    ##Mean-center and normalize
    #Y = Y - np.mean(Y, 1)[:, None]
    #Y = Y/np.sqrt(np.sum(Y**2, 1))[:, None]

    #PDs = doRipsFiltration(Y, 1)
    #pca = PCA()
    #Z = pca.fit_transform(Y)
    #print pca.explained_variance_
    #indices = range(len(Z))

    #plt.figure(figsize=(12, 6))
    #plt.subplot(121)
    #plt.title("2D PCA Raw Audio Embedding")
    #plt.scatter(Z[:, 0], Z[:, 1], c=indices)
    #plt.subplot(122)
    #plotDGM(PDs[1])
    #plt.title("Persistence Diagram")


    #Step 2: Do sliding window on audio novelty functions
    #(sliding window of sliding windows!)
    hopSize = 512

    #First do audio novelty function on music
    novFnMusic = getAudioNovelty(XMusic, FsMusic, hopSize)
    (MFCC, MFCCDCT, P) = getMFCC(XMusic, FsMusic, hopSize)
    dim = 20
    #Make sure the window size is half of a second, noting that
    #the audio novelty function has been downsampled by a "hopSize" factor
    Tau = (FsMusic /2)/(float(hopSize)*dim)
    dT = 1
    print len(MFCC)
    print len(MFCCDCT)
    Y = getSlidingWindowInteger(MFCCDCT[3], dim, Tau, dT)
    print("Y.shape = ", Y.shape)
    #Mean-center and normalize
    Y = Y - np.mean(Y, 1)[:, None]
    Y = Y/np.sqrt(np.sum(Y**2, 1))[:, None]

    PDs = doRipsFiltration(Y, 1)
    print PDs[1]
    print binPD(PDs[1], 2, 2, 0.5, 0.5, 6)
    pca = PCA()
    Z = pca.fit_transform(Y)
    print pca.explained_variance_
    indices = range(len(Z))

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title("2D PCA Music Novelty Function Sliding Window")
    plt.scatter(Z[:, 0], Z[:, 1], c=indices)
    plt.subplot(122)
    plotDGM(PDs[1])
    plt.title("Persistence Diagram")


    ##Now do audio novelty function on speech
    #novFnSpeech = getAudioNovelty(XSpeech, FsSpeech, hopSize)
    #dim = 20
    ##Make sure the window size is half of a second, noting that
    ##the audio novelty function has been downsampled by a "hopSize" factor
    #Tau = (FsSpeech)/(float(hopSize)*(dim/2))
    #dT = 1
    #Y = getSlidingWindowInteger(novFnSpeech, dim, Tau, dT)
    #print("Y.shape = ", Y.shape)
    ##Mean-center and normalize
    #Y = Y - np.mean(Y, 1)[:, None]
    #Y = Y/np.sqrt(np.sum(Y**2, 1))[:, None]

    #PDs = doRipsFiltration(Y, 1)
    #pca = PCA()
    #Z = pca.fit_transform(Y)

    ##plt.figure(figsize=(12, 6))
    ##plt.subplot(121)
    ##plt.title("2D PCA Speech Novelty Function Sliding Window")
    ##plt.scatter(Z[:, 0], Z[:, 1])
    ##plt.subplot(122)
    ##plotDGM(PDs[1])
    ##plt.title("Persistence Diagram")
    plt.show()
