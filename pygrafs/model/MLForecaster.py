import cPickle as pickle
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from collections import OrderedDict

class MLForecaster(object):
    def __init__(self, data_path, data_format, input_columns, output_column):
        self.data_path = data_path
        self.data_format = data_format
        self.input_columns = input_columns
        self.output_column = output_column
        self.models = OrderedDict()
        self.all_data = None
        return

    def load_data(self, exp="", query=None):
        data_files = sorted(glob(self.data_path + "*" + exp + "*" + self.data_format))
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

    def load_model(self, model_file):
        model_name = model_file.split('/')[-1].replace(".pkl","")
        with open(model_file) as model_file_obj:
            self.models[model_name] = pickle.load(model_file_obj)
        return

    def make_predictions(self):
        predictions = np.zeros((self.all_data.shape[0], len(self.models)))
        i = 0
        for model_name, model in self.models.iteritems():
            predictions[:, i] = model.predict(self.all_data.ix[:,self.input_columns])
        return predictions

