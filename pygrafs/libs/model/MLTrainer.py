from glob import glob
import cPickle
import numpy as np
import pandas as pd
import sys


class MLTrainer(object):
    """
    Handles loading of data files and training machine learning models on the data.

    :param data_path: path to data files.
    :param data_format: format of data files. Csv is the only currently supported format.
    :param input_columns: list of column names being input to model
    :param output_column:
    :return:
    """
    def __init__(self, data_path, data_format, input_columns, output_column):
        self.data_path = data_path
        self.data_format = data_format
        self.input_columns = input_columns
        self.output_column = output_column
        self.models = {}
        self.all_data = None
        return

    def load_data_files(self, exp="", query=None):
        """
        Loads data files from the specified directory.

        :param exp: Expression specifying which file names are are loaded.
        :param query: Pandas query string to filter data by column values
        :return:
        """
        data_files = sorted(glob(self.data_path + "*" + exp + "*" + self.data_format))
        data_file_list = []
        for data_file in data_files:
            sys.stdout.write("\rLoad File: " + data_file)
            sys.stdout.flush()
            if self.data_format == "csv":
                data_file_list.append(pd.read_csv(data_file))
            elif self.data_format == "hdf":
                data_file_list.append(pd.read_hdf(data_file, "data"))
        self.all_data = pd.concat(data_file_list, ignore_index=True)
        if query is not None:
            for q in query:
                self.all_data = self.all_data.query(q)
            self.all_data.reset_index(drop=True, inplace=True)

        self.all_data = self.all_data.replace(np.nan, 0)

    def sub_sample_data(self, num_samples, method='random', replace=False):
        if method == 'random':
            indices = np.random.choice(self.all_data.shape[0], num_samples, replace=replace)
            sampled_data = self.all_data.ix[indices, :]
            sampled_data.reset_index(inplace=True)
        else:
            sampled_data = None
        return sampled_data

    def train_model(self, model_name, model_obj):
        """
        Trains a machine learning model and adds it to the collection of models associated with this data.

        :param model_name: (str) unique name of the model
        :param model_obj: scikit-learn or similarly formatted object.
        :return:
        """
        self.models[model_name] = model_obj
        self.models[model_name].fit(self.all_data.loc[:, self.input_columns],
                                    self.all_data.loc[:, self.output_column])
        return
    
    def cross_validate_model(self, model_name, model_obj, n_folds):
        """
        Performs an n-fold randomized cross-validation on the input data and
        returns a set of predictions.

        :param model_name: (str) long name of the model being validated.
        :param model_obj: scikit-learn or similarly formatted model.
        :param n_folds: (int) The number of folds.
        :return: pandas Dataframe containing the predictions and associated metadata.
        """
        random_indices = np.random.permutation(self.all_data.shape[0])
        predictions = np.zeros(self.all_data.shape[0])
        print(model_name)
        for f in range(n_folds):
            split_start = random_indices.size * f / n_folds
            split_end = random_indices.size * (f + 1) / n_folds
            test_indices = random_indices[split_start:split_end]
            train_indices = np.concatenate((random_indices[:split_start], random_indices[split_end:]))
            print("Fold {0:d} Train {1:d}, Test {2:d}".format(f, train_indices.shape[0], test_indices.shape[0]))
            model_obj.fit(self.all_data.ix[train_indices, self.input_columns],
                          self.all_data.ix[train_indices, self.output_column])
            predictions[test_indices] = model_obj.predict(self.all_data.ix[test_indices, self.input_columns])
            self.models[model_name] = model_obj
            self.show_feature_importance()
        return predictions

    def site_validation(self, model_names, model_objs, pred_columns, test_day_interval, seed=505):
        """
        Train model at random subset of sites and validate at holdout sites.

        :param model_names: List of model names
        :param model_objs: List of model objects
        :param pred_columns: Columns from all_data to be included in output prediction data frame
        :param split_day: day of year used to split training and testing data
        :return: predictions and metadata in data frame
        """
        np.random.seed(seed)
        all_sites = np.sort(self.all_data['station'].unique())
        shuffled_sites = np.random.permutation(all_sites)
        train_stations = shuffled_sites[:shuffled_sites.size / 2]
        test_stations = shuffled_sites[shuffled_sites.size/2:]
        train_data = self.all_data.loc[self.all_data['station'].isin(train_stations) &
                                       (self.all_data['day_of_year'] % test_day_interval != 0)]
        test_data = self.all_data.loc[self.all_data['station'].isin(test_stations) &
                                      (self.all_data['day_of_year'] % test_day_interval == 0)]
        predictions = test_data[pred_columns]
        for m, model_obj in enumerate(model_objs):
            print model_names[m]
            model_obj.fit(train_data.loc[:, self.input_columns], train_data.loc[:, self.output_column])
            predictions[model_names[m]] = model_obj.predict(test_data.loc[:, self.input_columns])
            self.models[model_names[m]] = model_obj
        self.show_feature_importance()
        return predictions

    def show_feature_importance(self, num_rankings=10):
        """
        Display the top features in order of importance.

        :param num_rankings: Number of rankings to display
        :return:
        """
        for model_name, model_obj in self.models.iteritems():
            if hasattr(model_obj, "feature_importances_"):
                scores = model_obj.feature_importances_
                rankings = np.argsort(scores)[::-1]
                print(model_name)
                for i, r in enumerate(rankings[0:num_rankings]):
                    print("{0:d}. {1}: {2:0.3f}".format(i + 1, self.input_columns[r], scores[r]))
    
    def save_models(self, model_path):
        """
        Save models to pickle files with same name as specified in config file

        :param model_path: Path to model output files
        :return:
        """
        for model_name, model_obj in self.models.iteritems():
            with open(model_path + model_name + ".pkl", "w") as model_file:
                cPickle.dump(model_obj, model_file, cPickle.HIGHEST_PROTOCOL)
        return
