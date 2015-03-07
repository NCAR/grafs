import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from collections import OrderedDict
from mpl_toolkits.basemap import Basemap
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fore", help="Forecast file")
    args = parser.parse_args()
    #input_data_path = "/d2/dgagne/merged_data/"
    forecast_file = args.fore
    #input_file = input_data_path + forecast_file.split("/")[-1]
    #input_data = load_input_data(input_file)
    grid_predictions = csv_to_grid(forecast_file)
    input_data = None
    plot(grid_predictions, input_data)
    return


def csv_to_grid(forecast_file, models=["av_dswrf_sfc_f", "Random Forest", "Linear Regression"]):
    if forecast_file[-3:] == "csv":
        forecast_data = pd.read_csv(forecast_file)
    else:
        forecast_data = pd.read_hdf(forecast_file, "predictions")
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
    if filename[-3:] == "csv":
        data = pd.read_csv(filename, parse_dates=['date'])
    else:
        data = pd.read_hdf(filename, "predictions")
    print data['date']
    return data.loc[:, ['lon', 'lat', 'row', 'col', 'forecast_hour', 'av_dswrf_sfc']]


def plot(grid_predictions, input_data=None):
    contours = np.arange(0, 810, 10)
    bmap = Basemap(projection='cyl',
                   resolution='l',
                   llcrnrlon=grid_predictions['lon'].min(),
                   llcrnrlat=grid_predictions['lat'].min(),
                   urcrnrlon=grid_predictions['lon'].max(),
                   urcrnrlat=grid_predictions['lat'].max())

    plt.figure(figsize=(10, 10 * bmap.aspect))
    bmap.drawcoastlines()
    bmap.drawstates()
    bmap.drawcountries()
    i = 0
    title_obj = plt.title("")
    for f, fh in enumerate(grid_predictions['forecast_hour']):
        for m, model in enumerate(grid_predictions['models']):
            print(model, fh)

            contour_obj = plt.contourf(grid_predictions['lon'], grid_predictions['lat'], grid_predictions[model][f],
                         contours, extend='max', cmap=plt.get_cmap('YlOrRd', 80))
            if input_data is not None:
                fh_index = input_data['forecast_hour'] == fh
                plt.scatter(input_data.loc[fh_index,'lon'],
                            input_data.loc[fh_index, 'lat'],
                            20,
                            input_data.loc[fh_index, 'av_dswrf_sfc'],
                            vmin=0, vmax=800,
                            cmap=plt.get_cmap('YlOrRd', 80))
            if i == 0:
                plt.colorbar()
            if model == "av_dswrf_sfc_f":
                model_name = "NAM"
            else:
                model_name = model
            plt.setp(title_obj, text=model_name + " GHI {0} F{1:02d}".format(pd.to_datetime(grid_predictions['dates'][f]).strftime('%y-%m-%d %H:%M'),fh))

            plt.savefig("/d2/dgagne/grafs_forecast_figures/" + model.replace(" ","_") + "_F{0:02d}.png".format(fh),bbox_inches='tight',dpi=200)
            i += 1
            for col in contour_obj.collections:
                plt.gca().collections.remove(col)
            plt.draw()

if __name__ == "__main__":
    main()
