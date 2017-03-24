from .MLTrainer import MLTrainer
from copy import deepcopy
import pickle
import numpy as np
import pandas as pd
from .gridding import nearest_neighbor, cressman
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.linear_model import LinearRegression
from .persistence import Persistence

__author__ = 'David John Gagne'


class MLSiteTrainer(MLTrainer):
    """
    Train site-based machine learning models.
    """
    def __init__(self, data_path, data_format, input_columns, output_column, site_id_column="station"):
        super(MLSiteTrainer, self).__init__(data_path, data_format, input_columns, output_column,
                                            site_id_column=site_id_column)

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
        """
        Display the feature importance rankings for trained models that have a feature_importances_ attribute.

        :param num_rankings: Number of variables to display
        :return:
        """
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
        """
        Save the trained model objects to pickle files.

        :param model_path:
        :return:
        """
        for model_name, site_models in self.models.iteritems():
            for site, model in site_models.iteritems():
                pickle_file = open(model_path + "{0}_{1}.pkl".format(model_name, site), "w")
                pickle.dump(model, pickle_file)
                pickle_file.close()

    def site_validation(self, model_names, model_objs, pred_columns, test_day_interval, seed=505,
                        y_name="lat", x_name="lon", interp_method="nearest",
                        run_date_col="run_date", forecast_hour_col="forecast_hour"):
        """
        Randomly splits the training data into training and testing sites, trains each model, and makes
        predictions at the testing sites.

        :param model_names: List of strings giving a name for each model.
        :param model_objs: List of machine learning model objects
        :param pred_columns: Columns from the training data to be included in the prediction data frame.
        :param test_day_interval: Spacing between days used for testing based on the day of the year.
        :param seed: integer used to seed the random number generator.
        :param y_name: Name of the y-coordinate column in the training data
        :param x_name: Name of the x-coordinate column in the training data
        :param interp_method: "nearest" for nearest neighbor interpolation or "weighted" for error-based weighting of
            training sites.
        :param run_date_col: training data column containing the date of the model run
        :return: predictions: a data frame containing metadata and predictions for each test site from each model
            train_station_locations: a data frame with the coordinates of each site used for training
        """
        np.random.seed(seed)
        all_sites = np.sort(self.all_data[self.site_id_column].unique())
        shuffled_sites = np.random.permutation(all_sites)
        train_stations = np.array(sorted(shuffled_sites[:shuffled_sites.size / 2]))
        test_stations = np.array(sorted(shuffled_sites[shuffled_sites.size/2:]))
        run_day_of_year = pd.DatetimeIndex(self.all_data[run_date_col]).dayofyear
        self.all_data["run_day_of_year"] = run_day_of_year
        train_data = self.all_data.loc[self.all_data[self.site_id_column].isin(train_stations) &
                                       (run_day_of_year % test_day_interval != 0)]
        train_station_locations = train_data.groupby(self.site_id_column).first()[[x_name, y_name]].reset_index()

        evaluation_data = self.all_data.loc[self.all_data[self.site_id_column].isin(train_stations) &
                                            (run_day_of_year % test_day_interval == 0)]
        test_data = self.all_data.loc[self.all_data[self.site_id_column].isin(test_stations) &
                                      (run_day_of_year % test_day_interval == 0)]
        test_station_locations = test_data.groupby(self.site_id_column).first()[[x_name, y_name]].reset_index()

        predictions = test_data[pred_columns]
        site_predictions = evaluation_data[pred_columns]
        persist = Persistence(self.output_column, 24, "/d2/dgagne/mesonet_nc_2/mesonet", (-106, -85), (25, 40))
        print("Pred shape:", predictions.shape)
        predictions.loc[:, "Persistence"] = persist.make_predictions(evaluation_data,
                                                                     test_data)["Persistence"].values
        print(predictions["Persistence"])
        for m, model_obj in enumerate(model_objs):
            print(model_names[m])
            self.models[model_names[m]] = {}
            for site_name in train_stations:
                print(site_name)
                site_data = train_data.loc[train_data[self.site_id_column] == site_name]
                self.models[model_names[m]][site_name] = deepcopy(model_obj)
                self.models[model_names[m]][site_name].fit(site_data.loc[:, self.input_columns].values,
                                                           site_data.loc[:, self.output_column].values)
                if interp_method in ["nearest", "cressman"]:
                    eval_site_data = evaluation_data.loc[evaluation_data[self.site_id_column] == site_name, self.input_columns]
                    eval_site_id = evaluation_data[self.site_id_column] == site_name
                    if model_names[m] == "Random Forest Median":
                        site_predictions.loc[eval_site_id, model_names[m]] = np.median(np.array([t.predict(eval_site_data.values) for t in self.models[model_names[m]][site_name].estimators_]).T, axis=1)
                    else:
                        site_predictions.loc[eval_site_id, model_names[m]] = self.models[model_names[m]][site_name].predict(
                            eval_site_data.values)
            if interp_method == "nearest":
                for day in np.unique(evaluation_data["run_day_of_year"].values):
                    print("Day", day)
                    for hour in np.unique(evaluation_data[forecast_hour_col].values):
                        pred_rows = (test_data["run_day_of_year"] == day) & \
                                    (test_data[forecast_hour_col] == hour)
                        eval_rows = (evaluation_data["run_day_of_year"] == day) & \
                                    (evaluation_data[forecast_hour_col] == hour)
                        if np.count_nonzero(pred_rows) > 0 and np.count_nonzero(eval_rows) > 0:
                            predictions.loc[pred_rows, model_names[m]] = \
                                nearest_neighbor(site_predictions.loc[eval_rows,
                                                 [x_name, y_name, model_names[m]]],
                                                 predictions.loc[pred_rows],
                                                 y_name, x_name)
            elif interp_method == "cressman":
                for day in np.unique(evaluation_data["run_day_of_year"].values):
                    print("Day", day)
                    for hour in np.unique(evaluation_data[forecast_hour_col].values):
                        pred_rows = (test_data["run_day_of_year"] == day) & \
                                    (test_data[forecast_hour_col] == hour)
                        eval_rows = (evaluation_data["run_day_of_year"] == day) & \
                                    (evaluation_data[forecast_hour_col] == hour)
                        if np.count_nonzero(pred_rows) > 0 and np.count_nonzero(eval_rows) > 0:
                            predictions.loc[pred_rows, model_names[m]] = cressman(site_predictions.loc[eval_rows,
                                                                                                       [x_name, y_name,
                                                                                                        model_names[m]
                                                                                                        ]],
                                                                                  predictions.loc[pred_rows],
                                                                                  y_name, x_name)
            elif interp_method == "weighted":
                train_predictions = pd.DataFrame(index=train_data.index, columns=train_stations, dtype=float)
                test_predictions = pd.DataFrame(index=test_data.index, columns=train_stations, dtype=float)

                train_errors = pd.DataFrame(index=train_stations, columns=train_stations, dtype=float)
                train_distances = pd.DataFrame(squareform(pdist(train_station_locations[[x_name, y_name]].values)),
                                               index=train_stations, columns=train_stations, dtype=float)
                train_test_distances = pd.DataFrame(cdist(test_station_locations[[x_name, y_name]].values,
                                                          train_station_locations[[x_name, y_name]].values),
                                                    index=test_stations, columns=train_stations, dtype=float)
                for site_name in train_stations:
                    train_predictions[site_name] = self.models[model_names[m]][site_name].predict(
                        train_data[self.input_columns])
                    test_predictions[site_name] = self.models[model_names[m]][site_name].predict(
                        test_data[self.input_columns])
                    for pred_site_name in train_stations:
                        idx = train_data[self.site_id_column] == pred_site_name
                        train_errors.loc[site_name, pred_site_name] = \
                            np.mean(np.power(train_predictions.loc[idx, pred_site_name] -
                                             train_data.loc[idx, self.output_column], 2))
                error_lr = LinearRegression()
                valid_errors = np.where(~np.isnan(train_errors.values.ravel()))
                error_lr.fit(train_distances.values.reshape(train_distances.size, 1)[valid_errors],
                             train_errors.values.ravel()[valid_errors])
                pred_error = error_lr.predict(train_test_distances.values.reshape(train_test_distances.size, 1))
                pred_error = pred_error.reshape(train_test_distances.shape)
                pred_weights = 1.0 / pred_error
                pred_weights /= np.tile(pred_weights.sum(axis=1), (pred_weights.shape[1], 1)).T
                pred_weight_df = pd.DataFrame(pred_weights, index=test_stations, columns=train_stations)
                predictions[model_names[m]] = \
                    np.sum(test_predictions * pred_weight_df.loc[test_data[self.site_id_column].values].values, axis=1)
        return predictions, train_station_locations

