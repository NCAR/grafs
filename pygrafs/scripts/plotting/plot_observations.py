import matplotlib

from pygrafs.libs.data import ObsSite

matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
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
    parser.add_argument("--out", default="/d2/dgagne/obs_figures/",help="Path where figures are written.")
    args = parser.parse_args()
    obs_dates = pd.date_range(args.start, args.end, freq="H")
    all_data = load_data(obs_dates, args.var, args.obs)
    plot_obs_maps(all_data, obs_dates, args.var, args.out)
    return


def load_data(dates, variable, path):
    days = np.unique(dates.date)
    all_data = None
    for day in days:
        print(day)
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

def plot_obs_maps(all_data, obs_dates, variable, out_path):
    plt.figure(figsize=(10, 6))
    lon_bounds = (all_data['lon'].min() - 1.0, all_data['lon'].max() + 1.0)
    lat_bounds = (all_data['lat'].min() - 1.0, all_data['lat'].max() + 1.0)
    bmap = Basemap(projection="cyl",
                   resolution='l',
                   llcrnrlon=lon_bounds[0],
                   llcrnrlat=lat_bounds[0],
                   urcrnrlat=lat_bounds[1],
                   urcrnrlon=lon_bounds[1])
    bmap.drawcoastlines()
    bmap.drawcountries()
    bmap.drawstates()
    title_obj = plt.title("")
    for d, date in enumerate(obs_dates):
        print("Plotting ", date)
        di = all_data['date'] == date
        scatter = plt.scatter(all_data.loc[di,'lon'],
                              all_data.loc[di,'lat'],
                              20,
                              all_data.loc[di,variable],
                              cmap="hot",
                              vmin=0,
                              vmax=800)
        if d == 0:
            plt.colorbar()
        plt.setp(title_obj, text="{0} Observations {1}".format(variable, date.strftime("%Y-%m-%d %H:%M")))
        plt.savefig(out_path + "{0}_obs_{1}.png".format(variable, date.strftime("%Y%m%d%H%M")),dpi=200, bbox_inches="tight")
        scatter.remove()
    return


def plot_obs_time_series():
    return

if __name__ == "__main__":
    main()
