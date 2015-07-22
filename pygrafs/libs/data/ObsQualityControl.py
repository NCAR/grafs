import pandas as pd
import numpy as np
from ObsSite import ObsSite
from pvlib.solarposition import get_solarposition
from pvlib.clearsky import haurwitz
from pvlib.irradiance import extraradiation
from pvlib.location import Location
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def main():
    from datetime import datetime
    obs_path = "/d2/dicast/nt/der_data/obs/int_obs/"
    lon_bounds = (-125, -114)
    lat_bounds = (32, 44)
    qc = ObsQualityControl(datetime(2015, 6, 5), datetime(2015, 6, 25), "av_dswrf_sfc",
                           obs_path, lon_bounds, lat_bounds)
    qc.load_data()
    print qc.data.columns
    print qc.station_data.columns
    qc.clearness_index()
    print qc.data['haurwitz_kt'].values
    print qc.data['clearness_index'].values
    valid_counts = qc.invalid_count()
    biases = qc.bias()
    plt.figure(figsize=(8, 12))
    bmap = Basemap(projection="cyl", resolution="i", llcrnrlon=lon_bounds[0],
                   llcrnrlat=lat_bounds[0], urcrnrlon=lon_bounds[1], urcrnrlat=lat_bounds[1], fix_aspect=False)
    bmap.drawcountries()
    bmap.drawcoastlines()
    bmap.drawstates()
    #plt.scatter(qc.station_data['lon'].values, qc.station_data["lat"].values, 20,
    #            valid_counts["invalid"] / valid_counts["total"] * 100, vmin=0, vmax=50, edgecolors="", cmap="YlOrRd")
    plt.scatter(qc.station_data['lon'].values, qc.station_data["lat"].values, 20,
                biases["kt_bias"].values, vmin=-0.5, vmax=0.5, cmap="RdBu_r")
    plt.grid()
    plt.title("Clearness Index Outside 0-1 Range Relative Frequency")
    plt.colorbar()
    plt.show()
    return


class ObsQualityControl(object):
    def __init__(self, start_date, end_date, variable, obs_path, lon_bounds, lat_bounds):
        self.start_date = start_date
        self.end_date = end_date
        self.dates = pd.DatetimeIndex(start=self.start_date, end=self.end_date, freq="1D")
        self.variable = variable
        self.obs_path = obs_path
        self.lon_bounds = lon_bounds
        self.lat_bounds = lat_bounds
        self.station_data = None
        self.data = None
        return

    def load_data(self):
        obs_data = []
        station_data = []
        for date in self.dates:
            print date
            data_file = self.obs_path + "{0}/int_obs.{0}.nc".format(date.date().strftime("%Y%m%d"))
            data_obj = ObsSite(data_file)
            data_obj.load_data(self.variable)
            data_obj.filter_by_location(self.variable, self.lon_bounds, self.lat_bounds)
            data_obj.close()
            if data_obj.data is not None:
                obs_data.append(data_obj.data[self.variable])
                station_data.append(data_obj.station_data)
        self.data = pd.concat(obs_data, ignore_index=True)
        self.station_data = pd.concat(station_data)
        self.station_data.drop_duplicates(inplace=True)

    def clearness_index(self):
        new_columns = ["zenith", "elevation", "azimuth", "apparent_zenith", "etr", "etr_cos", "clearness_index",
                       "haurwitz_ghi", "haurwitz_kt"]
        for new_column in new_columns:
            self.data[new_column] = np.zeros((self.data.shape[0]))
        for station in self.station_data.index:
            print station
            loc = Location(self.station_data.loc[station, "lat"],
                           self.station_data.loc[station, "lon"],
                           tz="UTC",
                           altitude=self.station_data.loc[station, "elev"])
            station_ix = self.data["station"] == station
            station_dates = pd.DatetimeIndex(self.data.loc[station_ix, "valid_date"])
            #station_dates = pd.DatetimeIndex(station_dates.shift(-1, freq="30Min"))
            sol_pos = get_solarposition(station_dates, loc)
            sol_pos["etr"] = extraradiation(station_dates, method="pyephem")
            sol_pos["etr_cos"] = sol_pos["etr"] * np.cos(np.radians(sol_pos["apparent_zenith"]))
            sol_pos.loc[sol_pos["apparent_zenith"] > 90, "etr_cos"] = 0
            sol_pos["clearness_index"] = self.data.loc[station_ix, self.variable].values / sol_pos["etr_cos"]

            sol_pos["haurwitz_ghi"] = haurwitz(sol_pos["apparent_zenith"])
            sol_pos["haurwitz_kt"] = sol_pos["haurwitz_ghi"] / \
                (sol_pos["etr"] * np.cos(np.radians(sol_pos["apparent_zenith"])))
            self.data.loc[station_ix, new_columns] = sol_pos[new_columns].values

    def invalid_count(self):
        valid_count = pd.DataFrame(data=np.zeros((self.station_data.shape[0], 3)),
                                   columns=["valid", "invalid", "total"])
        for s, station in enumerate(self.station_data.index):
            si = (self.data["station"] == station) & (self.data["elevation"] > 0)
            valid_count.loc[s, "valid"] = np.count_nonzero((self.data.loc[si, 'clearness_index'] >= 0) &
                                              (self.data.loc[si, 'clearness_index'] <= 1))
            valid_count.loc[s, "invalid"] = np.count_nonzero(si) - valid_count.loc[s, "valid"]
            valid_count.loc[s, "total"] = np.count_nonzero(si)
        valid_count.index = self.station_data.index
        return valid_count

    def bias(self):
        bias = pd.DataFrame(data=np.zeros((self.station_data.shape[0], 2)),
                            index=self.station_data.index,
                            columns=["kt_bias", "ghi_bias"])
        for s, station in enumerate(self.station_data.index):
            si = (self.data["station"] == station) & (self.data["elevation"] > 0)
            bias.loc[station, "kt_bias"] = np.mean(self.data.loc[si, 'clearness_index']
                                                   - self.data.loc[si, "haurwitz_kt"])
            bias.loc[station, "ghi_bias"] = np.mean(self.data.loc[si, "av_dswrf_sfc"] - self.data.loc[si, "haurwitz_ghi"])
        return bias

if __name__ == "__main__":
    main()