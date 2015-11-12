from MLTrainer import MLTrainer
from copy import deepcopy
import cPickle
import numpy as np
import pandas as pd
from gridding import nearest_neighbor


__author__ = 'David John Gagne'


class MLSiteTrainer(MLTrainer):
    def __init__(self, data_path, data_format, input_columns, output_column, site_id_column="station"):
        self.site_id_column = site_id_column
        super(MLSiteTrainer, self).__init__(data_path, data_format, input_columns, output_column)

    def train_model(self, model_name, model_obj):
        """
        Trains a machine learning model at each specified site.

        :param model_name: (str) Name of the machine learning model
        :param model_obj: scikit-learn-style machine learning model object
        """
        self.models[model_name] = {}
        sites = np.unique(self.all_data[self.site_id_column].values)
        for site_name in sites:
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

    def site_validation(self, model_names, model_objs, pred_columns, test_day_interval, seed=505,
                        y_name="lat", x_name="lon"):
        np.random.seed(seed)
        all_sites = np.sort(self.all_data['station'].unique())
        shuffled_sites = np.random.permutation(all_sites)
        train_stations = shuffled_sites[:shuffled_sites.size / 2]
        test_stations = shuffled_sites[shuffled_sites.size/2:]
        run_day_of_year = pd.DatetimeIndex(self.all_data["run_date"]).dayofyear
        self.all_data["run_day_of_year"] = run_day_of_year
        train_data = self.all_data.loc[self.all_data['station'].isin(train_stations) &
                                       (run_day_of_year % test_day_interval != 0)]
        evaluation_data = self.all_data.loc[self.all_data['station'].isin(train_stations) &
                                            (run_day_of_year % test_day_interval == 0)]
        test_data = self.all_data.loc[self.all_data['station'].isin(test_stations) &
                                      (run_day_of_year % test_day_interval == 0)]
        predictions = test_data[pred_columns]
        site_predictions = evaluation_data[pred_columns]
        for m, model_obj in enumerate(model_objs):
            print model_names[m]
            self.models[model_names[m]] = {}
            for site_name in train_stations:
                print site_name
                site_data = train_data.loc[train_data[self.site_id_column] == site_name]
                self.models[model_names[m]][site_name] = deepcopy(model_obj)
                self.models[model_names[m]][site_name].fit(site_data.loc[:, self.input_columns],
                                                           site_data.loc[:, self.output_column])
                eval_site_data = evaluation_data.loc[evaluation_data[self.site_id_column] == site_name]
                site_predictions.loc[evaluation_data[self.site_id_column] == site_name,
                                     model_names[m]] = self.models[model_names[m]][site_name].predict(
                    eval_site_data.loc[:, self.input_columns])
            for day in np.unique(evaluation_data["run_day_of_year"].values):
                print day
                for hour in np.unique(evaluation_data["forecast_hour"].values):
                    pred_rows = (test_data["run_day_of_year"] == day) & (test_data["forecast_hour"] == hour)
                    eval_rows = (evaluation_data["run_day_of_year"] == day) & (evaluation_data["forecast_hour"] == hour)
                    interp_preds = nearest_neighbor(site_predictions.loc[eval_rows,
                                                                                  [x_name, y_name, model_names[m]]],
                                                                                  predictions.loc[pred_rows],
                                                                                  y_name, x_name)
                    predictions.loc[pred_rows, model_names[m]] = nearest_neighbor(site_predictions.loc[eval_rows,
                                                                                  [x_name, y_name, model_names[m]]],
                                                                                  predictions.loc[pred_rows],
                                                                                  y_name, x_name)
        train_station_locations = train_data[["station", x_name, y_name]].drop_duplicates()
        return predictions, train_station_locations

