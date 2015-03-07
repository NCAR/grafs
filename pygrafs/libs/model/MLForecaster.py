import cPickle as pickle
from netCDF4 import Dataset, date2num
import numpy as np
import pandas as pd
from collections import OrderedDict
import sys
from glob import glob


class MLForecaster(object):
    """
    Loads a trained machine learning model and generate a forecast.

    :param data_path: Path to data files.
    :param data_format: format of the input data
    :param input_columns: list of columns to be used as input for the machine learning model.
    :param output_column: name of the column being predicted
    """
    def __init__(self, data_path, data_format, input_columns, output_column):
        self.data_path = data_path
        self.data_format = data_format
        self.input_columns = input_columns
        self.output_column = output_column
        self.models = OrderedDict()
        self.all_forecasts = OrderedDict()
        return

    def load_data(self, exp=""):
        """
        Load forecasts from file and store them in all_forecasts organized by their input file

        :param exp: expression common to all files being loaded from a directory
        """
        forecast_files = sorted(glob(self.data_path + "*" + exp + "*" + self.data_format))
        for forecast_file in forecast_files:
            sys.stdout.write("\rLoad File: " + forecast_file)
            sys.stdout.flush()
            if self.data_format == "csv":
                self.all_forecasts[forecast_file] = pd.read_csv(forecast_file)
                self.all_forecasts[forecast_file].replace(np.nan, 0)
            elif self.data_format == "hdf":
                self.all_forecasts[forecast_file] = pd.read_hdf(forecast_file, "data")
                self.all_forecasts[forecast_file].replace(np.nan, 0)

    def load_model(self, model_file):
        """
        Load trained machine learning model from pickle file.

        :param model_file: Name of the model file
        """
        model_name = model_file.split('/')[-1].replace(".pkl","")
        with open(model_file) as model_file_obj:
            self.models[model_name] = pickle.load(model_file_obj)
        return

    def make_predictions(self, pred_columns, pred_path, pred_format, units=None):
        """
        Apply forecast data to the machine learning models and generate predictions

        :param pred_columns: Columns from input data to be included as meta data in output files
        :param pred_path: Path to directory containing prediction files.
        :param pred_format: format of prediction output files
        :return:
        """
        for forecast_file, forecast_data in self.all_forecasts.iteritems():
            sys.stdout.write("\rMake Predictions: " + forecast_file.split("/")[-1])
            sys.stdout.flush()
            predictions = forecast_data.loc[:, pred_columns]
            for model_name, model in self.models.iteritems():
                predictions[model_name] = model.predict(forecast_data.ix[:, self.input_columns])
            if pred_format.lower() == "csv":
                filename = pred_path + forecast_file.split("/")[-1]
                predictions.to_csv(filename, index=False, float_format="%0.3f")
            elif pred_format.lower() == "hdf":
                filename = pred_path + forecast_file.split("/")[-1]
                predictions.to_hdf(filename, "predictions", mode="w", complevel=4, complib="zlib")
            elif pred_format.lower() in ["nc", "netcdf"]:
                filename = pred_path + forecast_file.split("/")[-1].replace(".csv", ".nc")
                output = Dataset(filename, mode="w")
                grid_predictions, coordinates = self.predictions_to_grid(predictions)
                output.createDimension('time', coordinates['forecast_hour'].size)
                output.createDimension('y', coordinates['lon'].shape[0])
                output.createDimension('x', coordinates['lat'].shape[1])
                fh = output.createVariable('forecast_hour', 'i4', ('time',))
                fh[:] = coordinates['forecast_hour']
                for coord in ['lon', 'lat']:
                    cvar = output.createVariable(coord, 'f4', ('y', 'x'))
                    cvar[:] = coordinates[coord]
                for model_name, preds in grid_predictions.iteritems():
                    mod = output.createVariable(model_name, 'f4', ('time', 'y', 'x'))
                    mod[:] = preds
                    if units is not None:
                        mod.units = units
                output.close()
        return

    def predictions_to_grid(self, predictions):
        """
        Convert flat DataFrame of predictions to numpy array grids.

        :param predictions:
        :return:
        """
        forecast_hours = predictions['forecast_hour'].unique()
        row_size = predictions['row'].max() + 1
        col_size= predictions['col'].max() + 1
        grid_predictions = {}
        model_names = self.models.keys()
        for model_name in model_names:
            grid_predictions[model_name] = np.zeros(forecast_hours.size, row_size, col_size)
            grid_predictions[model_name][predictions['forecast_hour'].values,
                                         predictions['row'].values,
                                         predictions['col'].values] = predictions[model_name]
        fhz = predictions['forecast_hour'] == forecast_hours[0]
        coordinates = {}
        coordinates['forecast_hour'] = forecast_hours
        for coord in ['lon', 'lat']:
            coordinates[coord] = np.zeros(row_size, col_size)
            coordinates[coord][predictions['row'].values, predictions['col'].values] = predictions[coord]
        return grid_predictions, coordinates