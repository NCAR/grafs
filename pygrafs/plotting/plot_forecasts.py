import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from collections import OrderedDict
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fore", help="Forecast file")
    args = parser.parse_args()
    input_data_path = "/d2/dgagne/merged_data/"
    forecast_file = args.fore
    input_file = input_data_path + forecast_file.split("/")[-1]
    input_data = load_input_data(input_file)
    grid_predictions = csv_to_grid(forecast_file)
    plot(grid_predictions, input_data)
    return


def csv_to_grid(forecast_file, models=["av_dswrf_sfc_f", "Random Forest", "Linear Regression"]):
    forecast_data = pd.read_csv(forecast_file)
    forecast_hours = forecast_data['forecast_hour'].unique()
    dates = forecast_data['date'].unique()
    forecast_hours.sort()
    grid_shape = (forecast_hours.size, forecast_data['row'].max() + 1, forecast_data['col'].max() + 1)
    grid_predictions = OrderedDict()
    for model in models:
        grid_predictions[model] = np.zeros(grid_shape)
        grid_predictions[model][forecast_data['forecast_hour'].values,
                                forecast_data['row'].values,
                                forecast_data['col'].values] = forecast_data[model].values
    for index in ['lon','lat']:
        grid_predictions[index] = np.zeros(grid_shape[1:])
        grid_predictions[index][forecast_data['row'].values,
                                forecast_data['col'].values] = forecast_data[index].values
    grid_predictions['forecast_hour'] = forecast_hours
    grid_predictions['models'] = models
    grid_predictions['dates'] = pd.TimeSeries(dates)
    print grid_predictions['dates'].dtype
    return grid_predictions

def load_input_data(filename):
    data = pd.read_csv(filename, parse_dates=['date'])
    print data['date']
    return data.loc[:, ['lon', 'lat', 'row', 'col', 'forecast_hour', 'av_dswrf_sfc']]


def plot(grid_predictions, input_data=None):
    contours = np.arange(0, 810, 10)
    for f, fh in enumerate(grid_predictions['forecast_hour']):
        for m, model in enumerate(grid_predictions['models']):
            print(model, fh)
            plt.figure(figsize=(10, 8))
            plt.contourf(grid_predictions['lon'], grid_predictions['lat'], grid_predictions[model][f],
                         contours, extend='max', cmap=plt.get_cmap('YlOrRd', 80))
            if input_data is not None:
                fh_index = input_data['forecast_hour'] == fh
                plt.scatter(input_data.loc[fh_index,'lon'],
                            input_data.loc[fh_index, 'lat'],
                            20,
                            input_data.loc[fh_index, 'av_dswrf_sfc'],
                            vmin=0, vmax=800,
                            cmap=plt.get_cmap('YlOrRd', 80))
            plt.ylim(38, 39)
            plt.xlim(-121.75,-120.75)
            plt.colorbar()
            if model == "av_dswrf_sfc_f":
                model_name = "NAM"
            else:
                model_name = model
            plt.title(model_name + " GHI {0} F{1:02d}".format(pd.to_datetime(grid_predictions['dates'][f]).strftime('%y-%m-%d %H:%M'),fh))
            plt.savefig("/d2/dgagne/grafs_forecast_figures/" + model.replace(" ","_") + "_F{0:02d}.png".format(fh),bbox_inches='tight',dpi=200)
            plt.close()

if __name__ == "__main__":
    main()
