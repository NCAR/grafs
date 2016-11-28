from MLForecaster import MLForecaster
import cPickle
from glob import glob
import numpy as np
import sys


__author__ = 'David John Gagne'


class MLSiteForecaster(MLForecaster):
    def __init__(self,site_id_column, data_path, data_format, input_columns, output_column):
        self.site_id_column = site_id_column
        super(MLSiteForecaster, self).__init__(data_path, data_format, input_columns, output_column)

    def load_model(self, model_path, model_name):
        model_files = sorted(glob(model_path + model_name + "_*.pkl"))
        if len(model_files) > 0:
            self.models[model_name] = {}
            for model_file in model_files:
                model_site = model_file.split("/")[-1][:-4].split("_")[1]
                with open(model_file) as pickle_obj:
                    self.models[model_name][model_site] = cPickle.load(pickle_obj)

    def make_predictions(self, pred_columns):
        all_predictions = {}
        for forecast_file, forecast_data in self.all_forecasts.items():
            sys.stdout.write("\rMake Predictions: " + forecast_file.split("/")[-1])
            sys.stdout.flush()
            predictions = forecast_data.loc[:, pred_columns]
            for model_name, site_models in self.models.items():
                predictions[model_name] = np.zeros(predictions.shape[0])
                for site in sorted(list(site_models.keys())):
                    site_rows = forecast_data[self.site_id_column] == site
                    predictions.loc[site_rows,
                                    model_name] = site_models[site].predict(forecast_data.loc[site_rows,
                                                                                              self.input_columns])
            all_predictions[forecast_file] = predictions
        return all_predictions

    def predictions_to_grid(self, all_predictions, interp_method, grid_coordinates, time_name, y_name, x_name):
        y_size = grid_coordinates[y_name].shape[0]
        if len(grid_coordinates[x_name].shape) == 2:
            x_size = grid_coordinates[x_name].shape[1]
        else:
            x_size = grid_coordinates[x_name].shape[0]
        grid_predictions = {}
        model_names = self.models.keys()
        for model_name in model_names:
            grid_predictions[model_name] = np.zeros((grid_coordinates[time_name].size, y_size, x_size))
            model_cols = [y_name, x_name, model_name]
            for t, time in enumerate(grid_coordinates[time_name]):
                hour_predictions = all_predictions[model_name].loc[all_predictions[model_name][time_name] == time,
                                                                   model_cols]
                grid_predictions[model_name, t] = interp_method(hour_predictions, grid_coordinates, y_name, x_name)
        return grid_predictions




