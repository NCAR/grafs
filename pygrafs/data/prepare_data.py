from multiprocessing import Pool
from pygrafs.util.pool_manager import pool_manager
from glob import glob
import argparse
from datetime import datetime,timedelta

import numpy as np
import pandas as pd

from ModelGrid import ModelGrid
from ObsSite import ObsSite
from pygrafs.util.Config import Config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config",help="Configuration file")
    parser.add_argument("--proc","-p",default=1,type=int,help="Number of processors")
    args = parser.parse_args()
    config = Config(args.config)
    start_date = datetime.strptime(config.start_date,config.date_format)
    end_date = datetime.strptime(config.end_date,config.date_format)
    curr_date = start_date
    obs_total = 0
    if args.proc == 1:
        while curr_date <= end_date:
            print curr_date
            create_forecast_data(config, curr_date)
            curr_date += timedelta(days=1) 
    if args.proc > 1:
        pool = Pool(args.proc)
        procs = {}
        try:
            while curr_date < end_date:
                procs[curr_date] = pool.apply_async(create_forecast_data, (config, curr_date))
                curr_date += timedelta(days=1)
            pool_manager(procs, False)
        finally:
            pool.close()
            pool.join()
    return


def create_forecast_data(config, date):
    """
    Extract gridded forecast data and find matching observations at those locations.

    :param config: Config object containing parameters describing the model 
        and observation data.
    :param date: datetime object with the date of the model runs being used.
    """
    model_subset_grids, valid_datetimes = load_model_forecasts(config, date)
    model_unique_dates = None
    for model_file in valid_datetimes.iterkeys():
        print model_file
        if model_unique_dates is None:
            model_unique_dates = model_subset_grids[model_file].values()[0].get_unique_dates()
        else:
            model_unique_dates = np.union1d(model_unique_dates,
                                            model_subset_grids[model_file].values()[0].get_unique_dates())
    if config.mode == "train" and model_unique_dates is not None:
        all_obs = load_obs(config, model_unique_dates)
        match_model_obs(model_subset_grids, all_obs, config)

    return


def load_obs(config, dates):
    """
    Load site observations from every file in a particular date range. Assumes 1 netCDF file per day.

    :param config: Config object containing observation directory and variable information.
    :param dates: list of date or datetime objects for each day in the dataset.
    :return: dictionary of ObsSite objects containing observation data.
    """
    all_obs = {}
    for date in dates:
        obs_files = sorted(glob(config.obs_dir + date.strftime("/%Y%m%d/*.nc")))
        obs_total = 0
        if len(obs_files) > 0:
            all_obs[date] = ObsSite(obs_files[0])
            all_obs[date].load_data(config.obs_var)
            all_obs[date].close()
    return all_obs


def load_model_forecasts(config, date):
    """
    Loads gridded model output from netCDF files and stores it by variable and model run.

    :param config: Config object containing model directory, variable, and location information.
    :param date: date or datetime object containing the date of the model runs.
    """
    model_files = sorted(glob(config.model_dir + date.strftime("/%Y%m%d/*.nc")))
    model_subset_grids = {}
    valid_datetimes = {}
    if len(model_files) > 0:
        for model_file in model_files:
            print model_file
            model_grid = ModelGrid(model_file)
            model_subset_grids[model_file] = {}
            valid_datetimes[model_file] = {}
            for var in config.model_vars:
                model_subset_grids[model_file][var] = model_grid.load_subset(var,
                                                                config.t_range,
                                                                config.y_range,
                                                                config.x_range,
                                                                time_subset_type=config.time_subset_type,
                                                                space_subset_type=config.space_subset_type)
                valid_datetimes[model_file][var] = model_subset_grids[model_file][var].get_valid_data_times()         
                if "all" not in valid_datetimes[model_file].keys():
                    valid_datetimes[model_file]["all"] = valid_datetimes[model_file][var]
                else:
                    valid_datetimes[model_file]["all"] = np.intersect1d(valid_datetimes[model_file]["all"],
                                                                        valid_datetimes[model_file][var])
            model_grid.close()
    else:
        print config.model_dir + date.strftime("/%Y%m%d/*.nc")
        print model_files
    return model_subset_grids, valid_datetimes


def match_model_obs(model_grids, all_obs, config):
    """
    Match models output at observation sites with observations.

    :param model_grids: collection of model subset grids from load_model_forecasts.
    :param all_obs: collection of observations from load_obs.
    :param config: Config object with information about how model and observation data should be matched.
    """
    if config.match_method == "nearest":
        for model_name, model_grid in model_grids.iteritems():
            merged_data = None
            model_vars = sorted(model_grid.keys())
            unique_dates = model_grid[model_vars[0]].get_unique_dates()
            date_steps = np.array([t.date() for t in model_grid[model_vars[0]].times])
            for obs_date in sorted(all_obs.keys()):
                if obs_date in unique_dates:
                    station_indices = np.zeros((all_obs[obs_date].station_data.shape[0], 2), dtype=int)
                    station_coords = all_obs[obs_date].station_data.loc[:, ['lon', 'lat']].values
                    for s in range(station_indices.shape[0]):
                        station_indices[s] = model_grid[model_vars[0]].coordinate_to_index(*station_coords[s])
                    match_time_indices = np.intersect1d(np.where(date_steps == obs_date)[0],
                                   model_grid[model_vars[0]].valid_time_indices)
                    for mt in match_time_indices:
                        vt = model_grid[model_vars[0]].times[mt]
                        merged_data_step = pd.DataFrame({'station':all_obs[obs_date].station_data.index})
                        merged_data_step['date'] = vt
                        merged_data_step['valid_hour_utc'] = vt.hour
                        merged_data_step['valid_hour_pst'] = (vt.hour - 8) % 24
                        merged_data_step['forecast_hour']  = int((vt - model_grid[model_vars[0]].valid_times[0]).total_seconds() / 3600)
                        merged_data_step['day_of_year'] = vt.timetuple().tm_yday
                        merged_data_step['sine_doy'] = np.sin(vt.timetuple().tm_yday / 365.0 * np.pi)
                        for var in model_vars:
                            merged_data_step[var + "_f"] = model_grid[var].data[mt, station_indices[:,0],station_indices[:,1]]
                            neighbor_stats = np.array([model_grid[var].get_neighbor_stats(mt, station_indices[i, 0], station_indices[i, 1], indices=True, stats=config.stats) for i in range(station_indices.shape[0])])
                            for s, stat in enumerate(config.stats):
                                merged_data_step[var + "_f_" + stat] = neighbor_stats[:,s]
                        merged_data_obs = pd.merge(merged_data_step,all_obs[vt.date()].data[config.obs_var],
                                                   how="inner", on=['station', 'date'])
                        if merged_data is None:
                            merged_data = merged_data_obs
                        else:
                            merged_data = merged_data.append(merged_data_obs, ignore_index=True)
            merged_data = filter_data(merged_data, config.queries)
            if merged_data is not None:
                outfile = config.data_dir + model_name.split("/")[-1].strip(".nc") + ".csv"
                print "Writing " + outfile 
                merged_data.to_csv(outfile,
                                   float_format="%0.3f",
                                   index_label="record")
    return

def filter_data(data, queries):
    """
    Apply query strings to output DataFrames to filter out bad data points.

    :param data: DataFrame containing merged model and observation data.
    :param queries: list of query strings
    :return: filtered data with new indices.
    """
    for query in queries:
        data = data.query(query)
    data.reset_index(inplace=True)
    return data


if __name__ == "__main__":
    main()
