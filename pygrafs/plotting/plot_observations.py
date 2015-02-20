import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from pygrafs.data.ObsSite import ObsSite
from glob import glob
import numpy as np
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(prog="plot_observations")
    parser.add_argument("start", help="Start Date")
    parser.add_argument("end", help="End Date")
    parser.add_argument("-v", "--var", default="av_dswrf_sfc", help="Variable being plotted")
    parser.add_argument("--obs",
                        default="/d2/dicast/nt/der_data/obs/int_obs/",
                        help="Path to observation files")
    parser.add_argument("--out", help="Path where figures are written.")
    args = parser.parse_args()
    obs_dates = pd.date_range(args.start, args.end, freq="H")
    all_data = load_data(obs_dates, args.var, args.obs)
    print all_data
    return


def load_data(dates, variable, path):
    days = np.unique(dates.date)
    all_data = None
    for day in days:
        print day
        obs_files = sorted(glob(path + day.strftime("/%Y%m%d/*.nc")))
        if len(obs_files) > 0:
            obs_obj = ObsSite(obs_files[0])
            obs_obj.load_data(variable)
            obs_obj.close()
            if all_data is None:
                all_data = obs_obj.data[variable]
            else:
                all_data = all_data.append(obs_obj.data[variable], ignore_index=True)
    #all_data = all_data.ix[(all_data['date'] < dates[0]) & (all_data['date'] > dates[-1]), :]
    return all_data

def plot_obs_maps():
    return


def plot_obs_time_series():
    return

if __name__ == "__main__":
    main()