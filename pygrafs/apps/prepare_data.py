#!/usr/bin/env python
from multiprocessing import Pool
from glob import glob
import argparse
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pygrafs.libs.data.LandGrid import LandGrid
from pygrafs.libs.data.ModelGrid import ModelGrid
from pygrafs.libs.data.SolarData import make_solar_position_grid
from pygrafs.libs.data.ObsSite import ObsSite
from pygrafs.libs.util.Config import Config
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Configuration file")
    parser.add_argument("-p", "--proc", default=1, type=int, help="Number of processors")
    args = parser.parse_args()
    config = Config(args.config)
    start_date = datetime.strptime(config.start_date, config.date_format)
    end_date = datetime.strptime(config.end_date, config.date_format)
    curr_date = start_date
    if args.proc == 1:
        while curr_date <= end_date:
            print(curr_date)
            create_forecast_data(config, curr_date)
            curr_date += timedelta(days=1) 
    if args.proc > 1:
        pool = Pool(args.proc)
        procs = {}
        while curr_date <= end_date:
            procs[curr_date] = pool.apply_async(create_forecast_data, (config, curr_date))
            curr_date += timedelta(days=1)
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
    try:
        model_subset_grids, valid_datetimes = load_model_forecasts(config, date)
        model_unique_dates = None
        land_grids = None
        if hasattr(config, "land_files") and len(model_subset_grids) > 0:
            model_obj = model_subset_grids.values()[0].values()[0]
            land_grids = get_land_grid_data(config, model_obj.x, model_obj.y)
        for model_file in valid_datetimes.iterkeys():
            print(model_subset_grids[model_file].keys())
            if model_unique_dates is None:
                model_unique_dates = model_subset_grids[model_file].values()[0].get_unique_dates()
                print("Model unique dates", model_unique_dates)
            else:
                model_unique_dates = np.union1d(model_unique_dates,
                                                model_subset_grids[model_file].values()[0].get_unique_dates())
        if config.mode == "train" and model_unique_dates is not None:
            all_obs = load_obs(config, model_unique_dates)
            print all_obs
            match_model_obs(model_subset_grids, all_obs, config, land_grids=land_grids)

        if config.mode == "forecast" and model_unique_dates is not None:
            merge_model_forecasts(model_subset_grids, config)
    except Exception as e:
        print traceback.format_exc()
        raise e

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
        if len(obs_files) == 0:
            obs_files = sorted(glob(config.obs_dir + date.strftime("*.%Y%m%d.nc")))
        if len(obs_files) > 0:
            all_obs[date] = ObsSite(obs_files[0], meta_file=config.site_list_file)
            if type(config.obs_var) == str:
                obs_vars = [config.obs_var]
            else:
                obs_vars = config.obs_var
            for obs_var in obs_vars:
                all_obs[date].load_data(obs_var)
                all_obs[date].filter_by_location(obs_var, config.x_range, config.y_range)
                print("Num obs: {0:d}".format(all_obs[date].data[obs_var].shape[0]))
            all_obs[date].close()

    return all_obs


def load_model_forecasts(config, date):
    """
    Loads gridded model output from netCDF files and stores it by variable and model run.

    :param config: Config object containing model directory, variable, and location information.
    :param date: date or datetime object containing the date of the model runs.
    """
    model_files = sorted(glob(config.model_dir + date.strftime("/%Y%m%d/") + config.model_file_format))
    model_subset_grids = {}
    valid_datetimes = {}
    if len(model_files) > 0:
        for model_file in model_files:
            print("Loading " + model_file)
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
                valid_datetimes[model_file][var] = model_subset_grids[model_file][var].valid_times
                if "all" not in valid_datetimes[model_file].keys():
                    valid_datetimes[model_file]["all"] = valid_datetimes[model_file][var]
                else:
                    valid_datetimes[model_file]["all"] = np.intersect1d(valid_datetimes[model_file]["all"],
                                                                        valid_datetimes[model_file][var])
            model_grid.close()
            del model_grid
    else:
        print(config.model_dir + date.strftime("/%Y%m%d/*.nc"))
        print(model_files)
    return model_subset_grids, valid_datetimes


def get_land_grid_data(config, interp_lons, interp_lats):
    """
    Load static gridded data about each grid point, such as elevation and land cover information.

    :param config:
    :param interp_lons:
    :param interp_lats:
    :return:
    """
    land_grids = {}
    for l, land_file in enumerate(config.land_files):
        lg = LandGrid(land_file, data_vars=config.land_variables[l])
        for var in config.land_variables[l]:
            land_grids[var] = lg.interpolate_grid(var, interp_lons, interp_lats)
    return land_grids


def get_solar_grid_data(times, lon_grid, lat_grid, elevations=None):
    if elevations is None:
        elevations = np.zeros(lon_grid.shape)
    solar_positions = make_solar_position_grid(times, lon_grid, lat_grid, elevations)
    return solar_positions


def match_model_obs(model_grids, all_obs, config, land_grids=None):
    """
    Match models output at observation sites with observations.

    :param model_grids: collection of model subset grids from load_model_forecasts.
    :param all_obs: collection of observations from load_obs.
    :param config: Config object with information about how model and observation data should be matched.
    """
    if config.match_method == "nearest":
        # Loop through each model run and associated grid and match them with available observations.
        for model_name, model_grid in model_grids.iteritems():
            merged_data = None
            model_vars = sorted(model_grid.keys())
            unique_dates = model_grid[model_vars[0]].get_unique_dates()
            run_date = model_grid[model_vars[0]].init_time
            date_steps = np.array([t.date() for t in model_grid[model_vars[0]].times])
            print("Get Solar Positions")
            solar_data = get_solar_grid_data(model_grid[model_vars[0]].times,
                                             model_grid[model_vars[0]].x,
                                             model_grid[model_vars[0]].y)
            for obs_date in sorted(all_obs.keys()):
                print(obs_date)
                if obs_date in unique_dates:
                    station_indices = np.zeros((all_obs[obs_date].station_data.shape[0], 2), dtype=int)
                    station_coords = all_obs[obs_date].station_data.loc[:, ['lon', 'lat']].values
                    for s in range(station_indices.shape[0]):
                        station_indices[s] = model_grid[model_vars[0]].coordinate_to_index(*station_coords[s])
                    match_time_indices = np.intersect1d(np.where(date_steps == obs_date)[0],
                                                        model_grid[model_vars[0]].valid_time_indices)
                    for mt in match_time_indices:
                        vt = model_grid[model_vars[0]].times[mt]
                        merged_data_step = pd.DataFrame({'station': all_obs[obs_date].station_data.index})
                        # Create metadata columns
                        merged_data_step['run_date'] = run_date
                        merged_data_step['valid_date'] = vt
                        merged_data_step['valid_hour_utc'] = vt.hour
                        merged_data_step['valid_hour_cst'] = (vt.hour - 6) % 24
                        merged_data_step['forecast_hour'] = int((vt - run_date
                                                                 ).total_seconds() / 3600)
                        merged_data_step['day_of_year'] = vt.timetuple().tm_yday
                        merged_data_step['sine_doy'] = np.sin(vt.timetuple().tm_yday / 365.0 * np.pi)
                        merged_data_step['row'] = station_indices[:, 0]
                        merged_data_step['col'] = station_indices[:, 1]
                        if land_grids is not None:
                            for land_var in sorted(land_grids.keys()):
                                merged_data_step[land_var] = land_grids[land_var][station_indices[:, 0],
                                                                                  station_indices[:, 1]]
                        for var in model_vars:
                            # Extract grid point model values
                            merged_data_step[var + "_f"] = model_grid[var].data[mt,
                                                                                station_indices[:, 0],
                                                                                station_indices[:, 1]]
                            # Calculate neighborhood statistics from data and add it to data frame
                            neighbor_stats = model_grid[var].get_neighbor_grid_stats(mt,
                                                                                     stats=config.stats,
                                                                                     neighbor_radius=config.neighbor_radius)
                            for s, stat in enumerate(config.stats):
                                merged_data_step[var + "_f_" + stat] = neighbor_stats[s,
                                                                                      station_indices[:, 0],
                                                                                      station_indices[:, 1]]
                        for svar, sgrid in solar_data.iteritems():
                            merged_data_step[svar + "_f"] = sgrid.data[mt, station_indices[:, 0], station_indices[:, 1]]
                        merged_data_step["CLRI_f"] = model_grid[config.model_ghi_var].data[mt,
                                                                                           station_indices[:, 0],
                                                                                           station_indices[:, 1]] / \
                                                     solar_data["ETRC"].data[mt,
                                                                             station_indices[:, 0],
                                                                             station_indices[:, 1]]
                        # Match model output instances with observations
                        if type(config.obs_var) == str:
                            merged_data_obs = pd.merge(merged_data_step, all_obs[vt.date()].data[config.obs_var],
                                                       how="inner", on=['station', 'valid_date'])
                        else:
                            merged_data_obs = None
                            for obs_var in config.obs_var:
                                if merged_data_obs is None:
                                    merged_data_obs = pd.merge(merged_data_step, all_obs[vt.date()].data[obs_var],
                                                               how="inner", on=['station', 'valid_date'])
                                else:
                                    merged_data_obs = pd.merge(merged_data_obs, all_obs[vt.date()].data[obs_var].loc[:,
                                                                                ['station', 'valid_date', obs_var]],
                                                               how="inner", on=['station', 'valid_date'])
                        if merged_data is None:
                            merged_data = merged_data_obs
                        else:
                            merged_data = merged_data.append(merged_data_obs, ignore_index=True)
            if merged_data is not None:
                merged_data = filter_data(merged_data, config.queries)
                outfile = config.data_dir + config.model_name + "_" + \
                    model_name.split("/")[-1].replace(".nc", "." + config.out_format)
                print("Writing " + outfile)
                if config.out_format == "csv":
                    merged_data.to_csv(outfile,
                                       float_format="%0.3f",
                                       index_label="record")
                elif config.out_format == "hdf":
                    merged_data.to_hdf(outfile, "data", mode="w", complevel=4, complib="zlib")
    return


def merge_model_forecasts(model_grids, config, land_grids=None):
    """
    Given loaded model grids, merge the grids into a data table and output the table to csv.

    :param model_grids:
    :param config:
    :return:
    """
    for model_name, model_grid in model_grids.iteritems():
        merged_data = None
        model_vars = sorted(model_grid.keys())
        columns = ['run_date', 'valid_date', 'valid_hour_utc', 'valid_hour_pst', 'forecast_hour',
                    'day_of_year', 'sine_doy', 'row', 'col', 'lon', 'lat']
        for v in model_vars:
            columns.extend([v + '_f'] + [v + '_f_' + stat for stat in config.stats])
        time_indices = model_grid[model_vars[0]].valid_time_indices
        valid_times = model_grid[model_vars[0]].valid_times
        data_indices = np.indices(model_grid[model_vars[0]].data[0].shape)
        for t, time_step in enumerate(time_indices):
            prog = "*" * t + " " * (time_indices.size - t)
            #TODO use log library instead of print statements
            sys.stdout.write("\rTimestep: {0:02d}/{1:02d} [{2}]".format(time_step, time_indices.size, prog))
            sys.stdout.flush()
            merged_data_step = pd.DataFrame(index=np.arange(model_grid[model_vars[0]].data[0].size), columns=columns)
            vt = valid_times[t]
            merged_data_step['run_date'] = valid_times[0]
            merged_data_step['valid_date'] = vt
            merged_data_step['valid_hour_utc'] = vt.hour
            merged_data_step['valid_hour_pst'] = (vt.hour - 8) % 24
            merged_data_step['forecast_hour'] = int((vt - valid_times[0]).total_seconds() / 3600)
            merged_data_step['day_of_year'] = vt.timetuple().tm_yday
            merged_data_step['sine_doy'] = np.sin(vt.timetuple().tm_yday / 365.0 * np.pi)
            merged_data_step['row'] = data_indices[0].ravel()
            merged_data_step['col'] = data_indices[1].ravel()
            merged_data_step['lon'] = model_grid[model_vars[0]].x.ravel()
            merged_data_step['lat'] = model_grid[model_vars[0]].y.ravel()
            if land_grids is not None:
                for land_var in sorted(land_grids.keys()):
                    merged_data_step[land_var] = land_grids[land_var].ravel()
            for var in model_vars:
                merged_data_step.loc[:, var + "_f"] = model_grid[var].data[time_step].flatten()
                stat_arrays = model_grid[var].get_neighbor_grid_stats(time_step, stats=config.stats)
                for s, stat in enumerate(config.stats):
                    merged_data_step.loc[:, var + "_f_" + stat] = stat_arrays[s].flatten()
            if merged_data is None:
                merged_data = merged_data_step
            else:
                merged_data = merged_data.append(merged_data_step, ignore_index=True)
        print("\n")
        if merged_data is not None:
            merged_data = filter_data(merged_data, config.queries)
            outfile = config.data_dir + config.model_name + \
                model_name.split("/")[-1].replace(".nc", "." + config.out_format)
            print("Writing " + outfile)
            if config.out_format == "csv":
                merged_data.to_csv(outfile, float_format="%0.3f", index_label="record")
            elif config.out_format == "hdf":
                merged_data.to_hdf(outfile, "data", mode="w", complevel=4, complib="zlib")
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
    data.reset_index(drop=True, inplace=True)
    return data


if __name__ == "__main__":
    main()
