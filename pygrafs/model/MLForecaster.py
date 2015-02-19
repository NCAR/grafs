import cPickle as pickle
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from collections import OrderedDict
import sys
from glob import glob


class MLForecaster(object):
    def __init__(self, data_path, data_format, input_columns, output_column):
        self.data_path = data_path
        self.data_format = data_format
        self.input_columns = input_columns
        self.output_column = output_column
        self.models = OrderedDict()
        self.all_forecasts = OrderedDict()
        return

    def load_data(self, exp="", query=None):
        forecast_files = sorted(glob(self.data_path + "*" + exp + "*" + self.data_format))
        for forecast_file in forecast_files:
            sys.stdout.write("\rLoad File: " + forecast_file)
            sys.stdout.flush()
            if self.data_format == "csv":
                self.all_forecasts[forecast_file] = pd.read_csv(forecast_file)
                self.all_forecasts[forecast_file].replace(np.nan, 0)

    def load_model(self, model_file):
        model_name = model_file.split('/')[-1].replace(".pkl","")
        with open(model_file) as model_file_obj:
            self.models[model_name] = pickle.load(model_file_obj)
        return

    def make_predictions(self, pred_columns, pred_path, pred_format):
        for forecast_file, forecast_data in self.all_forecasts.iteritems():
            sys.stdout.write("\rMake Predictions: " + forecast_file.split("/")[-1])
            sys.stdout.flush()
            predictions = forecast_data.loc[:, pred_columns]
            for model_name, model in self.models.iteritems():
                predictions[model_name] = model.predict(forecast_data.ix[:, self.input_columns])
            if pred_format == "csv":
                filename = pred_path + forecast_file.split("/")[-1]
                predictions.to_csv(filename, index=False, float_format="%0.3f")
        return

