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

    def extract_features_from_wav(self, file):
        results = []
        FsMusic, XMusic = scipy.io.wavfile.read(file)
        for extractor, param_calculator in zip(self.extractors, self.param_calculators):
            raw_features = extractor(FsMusic, XMusic)
            (dim, tau, dt) = self.param_calculators(FsMusic, XMusic)
            Y = getSlidingWindowInteger(raw_features, dim, Tau, dT)
            print("Y.shape = ", Y.shape)
            #Mean-center and normalize
            Y = Y - np.mean(Y, 1)[:, None]
            Y = Y/np.sqrt(np.sum(Y**2, 1))[:, None]

            PDs = doRipsFiltration(Y, 1)
            results.push(ExtractedFeature(raw_features, PDs[1], binPD(PDs[1], 2, 2, 0, 0, 6)))
        return results





