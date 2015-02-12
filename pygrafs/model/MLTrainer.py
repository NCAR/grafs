from glob import glob
import cPickle as pickle
import numpy as np
import pandas as pd


class MLTrainer(object):
    def __init__(self, data_path, data_format, input_columns, output_column):
        self.data_path = data_path
        self.data_format = data_format
        self.input_columns = input_columns
        self.output_column = output_column
        self.models = {}
        self.all_data = None
        return

    def load_data_files(self,query=None):
        data_files = sorted(glob(self.data_path + "*.05." + self.data_format))
        data_file_list = []
        for data_file in data_files:
            print data_file
            if self.data_format == "csv":
                data_file_list.append(pd.read_csv(data_file))
        self.all_data = data_file_list[0].append(data_file_list[1:],ignore_index=True)
        if query is not None:
            self.all_data = self.all_data.query(query)
            self.all_data.reset_index(inplace=True)
        self.all_data = self.all_data.replace(np.nan, 0)

    def train_model(self, model_name, model_obj):
        self.models[model_name] = model_obj
        self.models[model_name].fit(self.all_data.loc[:, self.input_columns],
                                    self.all_data.loc[:, self.output_column])
        return
    
    def cross_validate_model(self, model_name, model_obj, n_folds):
        random_indices = np.random.permutation(self.all_data.shape[0])
        predictions = np.zeros(self.all_data.shape[0])
        print model_name
        for f in range(n_folds):
            split_start = random_indices.size * f / n_folds
            split_end = random_indices.size * (f + 1) / n_folds
            test_indices = random_indices[split_start:split_end]
            train_indices = np.concatenate((random_indices[:split_start], random_indices[split_end:]))
            print "Fold {0:d} Train {1:d}, Test {2:d}".format(f, train_indices.shape[0], test_indices.shape[0])
            print self.input_columns
            model_obj.fit(self.all_data.ix[train_indices, self.input_columns],
                          self.all_data.ix[train_indices, self.output_column])
            predictions[test_indices] = model_obj.predict(self.all_data.ix[test_indices, self.input_columns])
            self.models[model_name] = model_obj
            self.show_feature_importance()
        return predictions

    def show_feature_importance(self, num_rankings=10):
        for model_name, model_obj in self.models.iteritems():
            if hasattr(model_obj,"feature_importances_"):
                scores = model_obj.feature_importances_
                rankings = np.argsort(scores)[::-1]
                print model_name
                for i,r in enumerate(rankings[0:num_rankings]):
                    print "{0:d}. {1}: {2:0.3f}".format(i,self.input_columns[r],scores[r])
    
    def save_models(self, model_path):
        for model_name, model_obj in self.models.iteritems():
            with open(model_path + model_name + ".pkl", "w") as model_file:
                pickle.dump(model_obj, model_file)
        return
