from WavFeatureExtractor import *
from MusicFeatures import *

def e1(XMusic, FsMusic):
	(MFCC, MFCCDT, P) = getMFCC(XMusic, FsMusic, 512)
	return MFCCDT[3]

def p1(XMusic, FsMusic):
	return (20, ((FsMusic /2)/(float(512)*20)), 1)

extractor = WavFeatureExtractor([e1], [p1], 0, 0, 0)

print extractor.extract_features_from_wav('2017_03_30_ao.wav')
