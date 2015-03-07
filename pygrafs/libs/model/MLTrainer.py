from glob import glob
import cPickle as pickle
import numpy as np
import pandas as pd
import sys
from collections import OrderedDict


class MLTrainer(object):
    """
    Handles loading of data files and training machine learning models on the data.

    :param data_path: path to data files.
    :param data_format: format of data files. Csv is the only currently supported format.
    :param input_columns: list of column names being input to model
    :param output_column:
    :return:
    """
    def __init__(self, data_path, data_format, input_columns, output_column, diff_column=None):
        self.data_path = data_path
        self.data_format = data_format
        self.input_columns = input_columns
        self.output_column = output_column
        self.models = OrderedDict()
        self.diff_column = diff_column
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
        self.all_data = data_file_list[0].append(data_file_list[1:], ignore_index=True)
        if "level_0" in self.all_data.columns:
            self.all_data.drop("level_0", axis=1, inplace=True)
        if query is not None:
            for q in query:
                self.all_data = self.all_data.query(q)
            self.all_data.reset_index(inplace=True)

        self.all_data = self.all_data.replace(np.nan, 0)

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
        print model_name
        for f in range(n_folds):
            split_start = random_indices.size * f / n_folds
            split_end = random_indices.size * (f + 1) / n_folds
            test_indices = random_indices[split_start:split_end]
            train_indices = np.concatenate((random_indices[:split_start], random_indices[split_end:]))
            print("Fold {0:d} Train {1:d}, Test {2:d}".format(f, train_indices.shape[0], test_indices.shape[0]))
            model_obj.fit(self.all_data.ix[train_indices, self.input_columns],
                          self.all_data.ix[train_indices, self.output_column])
            predictions[test_indices] = model_obj.predict(self.all_data.ix[test_indices, self.input_columns])
            if self.diff_column is not None:
                predictions[test_indices] = self.all_data[self.diff_column].values[test_indices] - predictions[test_indices]
            self.models[model_name] = model_obj
            self.show_feature_importance()
        return predictions

    def show_feature_importance(self, num_rankings=10):
        """
        Display the top features in order of importance.

        :param num_rankings: Number of rankings to display
        :return:
        """
        for model_name, model_obj in self.models.iteritems():
            if hasattr(model_obj,"feature_importances_"):
                scores = model_obj.feature_importances_
                rankings = np.argsort(scores)[::-1]
                print(model_name)
                for i,r in enumerate(rankings[0:num_rankings]):
                    print("{0:d}. {1}: {2:0.3f}".format(i + 1, self.input_columns[r], scores[r]))
    
    def save_models(self, model_path):
        """
        Save models to pickle files with same name as specified in config file

        :param model_path: Path to model output files
        :return:
        """
        for model_name, model_obj in self.models.iteritems():
            with open(model_path + model_name + ".pkl", "w") as model_file:
                pickle.dump(model_obj, model_file)
        return
