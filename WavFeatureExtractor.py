from TDA import *
import scipy.io.wavfile
from ExtractedFeature import *
from SlidingWindow import *

class WavFeatureExtractor:

    def __init__(self, extractor_functions, parameter_calculators, default_dim, default_tau, default_dt):
        self.extractors = extractor_functions
        self.param_calculators = parameter_calculators
        self.def_dim = default_dim
        self.def_tau = default_tau
        self.def_dt = default_dt

    def binPD (self, PD, birth_max, death_max, birth_min=0.0, death_min=0.0, axis_bins=3):
        feature_vector = [0] * (axis_bins * axis_bins)
        for point in PD:
            birth_bin = int((point[0] - birth_min - 0.0001) / (float(birth_max - birth_min) / axis_bins))
            death_bin = int((point[1] - death_min - 0.0001) / (float(death_max - death_min) / axis_bins))
            #print (point, birth_bin, death_bin)

            feature_vector[(birth_bin * axis_bins) + death_bin] += 1

        return feature_vector

    def extract_features_from_wav(self, file):
        results = []
        FsMusic, XMusic = scipy.io.wavfile.read(file)
        for extractor, param_calculator in zip(self.extractors, self.param_calculators):
            raw_features = extractor(XMusic, FsMusic)
            (dim, tau, dT) = param_calculator(XMusic, FsMusic)
            #print (dim, tau, dT)
            Y = getSlidingWindowInteger(raw_features, dim, tau, dT)
            #print("Y.shape = ", Y.shape)
            #Mean-center and normalize
            Y = Y - np.mean(Y, 1)[:, None]
            Y = Y/np.sqrt(np.sum(Y**2, 1))[:, None]

            PDs = doRipsFiltration(Y, 1)
            results.append(ExtractedFeature(XMusic, raw_features, PDs[1], self.binPD(PDs[1], 2, 2, 0, 0, 6)))
        return results
