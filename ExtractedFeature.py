class ExtractedFeature:

    def __init__(self, raw_data, raw_feature_data, persistence_diagram_1d, binned_diagram_1d):
        self.features = {}
        self.features['raw_data'] = raw_data
        self.features['raw_feature_data'] = raw_feature_data
        self.features['pd_1d'] = persistence_diagram_1d
        self.features['binned_pd_1d'] = binned_diagram_1d
