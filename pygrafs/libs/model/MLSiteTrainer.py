from MLTrainer import MLTrainer
from copy import deepcopy
import cPickle
import numpy as np


__author__ = 'David John Gagne'


class MLSiteTrainer(MLTrainer):
    def __init__(self, site_info, site_id_column, data_path, data_format, input_columns, output_column):
        self.site_info = site_info
        self.site_id_column = site_id_column
        if self.site_id_column not in self.site_info.keys():
            raise IndexError(self.site_id_column + " not found in site_info")
        super(MLSiteTrainer, self).__init__(data_path, data_format, input_columns, output_column)

    def train_model(self, model_name, model_obj):
        """
        Trains a machine learning model at each specified site.

        :param model_name: (str) Name of the machine learning model
        :param model_obj: scikit-learn-style machine learning model object
        """
        self.models[model_name] = {}
        for site_name in self.site_info[self.site_id_column]:
            site_data = self.all_data.loc[self.all_data[self.site_id_column] == site_name]
            site_model = deepcopy(model_obj)
            site_model.fit(site_data[self.input_columns], site_data[self.output_column])
            self.models[model_name][site_name] = site_model

    def show_feature_importance(self, num_rankings=10):
        for model_name, site_models in self.models.iteritems():
            if hasattr(site_models.values()[0], "feature_importances_"):
                scores = np.zeros((len(site_models), len(self.input_columns)))
                for s, site in sorted(site_models.keys()):
                    scores[s] = site_models[site].feature_importances_
                rankings = np.argsort(scores.mean(axis=0))[::-1]
                print(model_name)
                for i, r in enumerate(rankings[0:num_rankings]):
                    print("{0:d}. {1}: {2:0.3f} {3:0.3f}".format(i + 1, self.input_columns[r], scores.mean(axis=0)[r],
                                                                 scores.std(axis=0)[r]))

    def save_models(self, model_path):
        for model_name, site_models in self.models.iteritems():
            for site, model in site_models.iteritems():
                pickle_file = open(model_path + "{0}_{1}.pkl".format(model_name, site), "w")
                cPickle.dump(model, pickle_file)
                pickle_file.close()




