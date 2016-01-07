import pandas as pd
from pvlib.solarposition import get_solarposition
from pvlib.irradiance import extraradiation
from pvlib.location import Location
import numpy as np
import os
from netCDF4 import Dataset, date2num, stringtochar
from datetime import datetime


class MesonetRawData(object):
    def __init__(self, start_date, end_date, stations, path, station_info_file):
        self.start_date = start_date
        self.end_date = end_date
        self.dates = pd.DatetimeIndex(start=self.start_date, end=self.end_date,
                                      freq="5min", tz="UTC")
        self.stations = stations
        self.data = {}
        self.averaged_data = {}
        self.path = path
        self.station_info = pd.read_csv(station_info_file, index_col="stid")
        self.units = dict(RELH="%",
                          TAIR="degrees Celsius",
                          WSPD="m s-1",
                          WVEC="m s-1",
                          WDIR="degrees",
                          WDSD="degrees",
                          WSSD="m s-1",
                          WMAX="m s-1",
                          RAIN="mm",
                          PRES="mb",
                          SRAD="W m-2",
                          TA9M="degrees Celsius",
                          WS2M="m s-1",
                          CLRI="",
                          ETRC="W m-2",
                          elevation="degrees",
                          azimuth="degrees",
                          zenith="degrees")

    def load_data(self):
        def read_mesonet_day(date):
            filename = self.path + "{0:d}/{1:02d}/{2:02d}/{0:d}{1:02d}{2:02d}{3}.mts".format(
                date.year, date.month, date.day, station.lower())
            data = pd.read_table(filename, sep="[ ]{1,5}", skiprows=2, engine="python",
                                 index_col=False,
                                 na_values=np.arange(-999, -989, 1).tolist() + ["   "])
            data.index = pd.DatetimeIndex(pd.Timestamp(date) + pd.TimedeltaIndex(data["TIME"], unit="m"), tz="UTC")
            return data
        all_days = np.unique(self.dates.date)
        for station in self.stations:
            station_data = pd.concat(map(read_mesonet_day, all_days))
            self.data[station] = station_data.loc[self.dates]

    def running_mean(self, variable, window_size, sample_frequency):
        mean_data = pd.DataFrame()
        for station in self.stations:
            mean_data[station] = pd.rolling_mean(self.data[station][variable], window_size)[
                sample_frequency::sample_frequency]
        self.averaged_data[variable + "_Mean"] = mean_data
        return mean_data

    def running_sd(self, variable, window_size, sample_frequency):
        sd_data = pd.DataFrame()
        for station in self.stations:
            sd_data[station] = pd.rolling_std(self.data[station][variable], window_size)[
                sample_frequency::sample_frequency]
        self.averaged_data[variable + "_SD"] = sd_data
        return sd_data

    def solar_data(self, radiation_var="SRAD"):
        columns = ["elevation", "azimuth", "zenith", "ETRC", "CLRI"]
        for station in self.stations:
            loc = Location(self.station_info.loc[station, "nlat"], self.station_info.loc[station, "elon"], tz="UTC",
                           altitude=self.station_info.loc[station, "elev"])
            solar_data = get_solarposition(self.data[station].index, loc)
            solar_data["EXTR"] = extraradiation(self.data[station].index, method="spencer")
            solar_data["ETRC"] = solar_data["EXTR"] * np.cos(np.radians(solar_data["zenith"]))
            solar_data.loc[solar_data["zenith"] > 90, "ETRC"] = 0
            solar_data["CLRI"] = np.zeros(solar_data["ETRC"].size)
            si = solar_data["ETRC"].values > 0
            solar_data["CLRI"][si] = self.data[station][radiation_var].values[si] / solar_data["ETRC"].values[si]
            solar_data["CLRI"][solar_data["ETRC"] == 0] = np.nan
            self.data[station][columns] = solar_data[columns]

    def averaged_data_to_netcdf(self, out_path, time_units="seconds since 1970-01-01 00 UTC", station_numbers=None):
        for variable in sorted(self.averaged_data.keys()):
            unique_dates = np.unique(self.averaged_data[variable].index.date)
            for date in unique_dates:
                filename = out_path + "mesonet.{0}.nc".format(date.strftime("%Y%m%d"))
                if os.access(filename, os.R_OK):
                    mode = "a"
                else:
                    mode = "w"
                ds = Dataset(filename, mode=mode)
                if mode == "w":
                    ds.createDimension("recNum", size=None)
                    ds.createDimension("timesPerDay", size=24)
                    ds.createDimension("stationNameSize", size=4)
                    c_time = ds.createVariable("creation_time", "f8")
                    c_time.assignValue(date2num(datetime.utcnow(), time_units))
                    c_time.units = time_units
                    c_time.long_name = "time at which file was created"
                    obs_times = ds.createVariable("time_nominal", "f8", ("timesPerDay", ))
                    obs_times.long_name = "observation time"
                    obs_times.units = time_units
                    valid_dates = self.averaged_data[variable].index[
                        self.averaged_data[variable].index.date == date].to_pydatetime()
                    obs_times[:] = date2num(valid_dates, time_units)
                    station_names = ds.createVariable("stationName", "S1", ("recNum", "stationNameSize"))
                    print self.averaged_data[variable].columns.values
                    station_names[:] = stringtochar(self.averaged_data[variable].columns.values.astype("S4"))
                    if station_numbers is not None:
                        site_list = ds.createVariable("site_list", "i4", ("recNum",))
                        site_list.long_name = "DICAST Identification Number"
                        site_list.reference = "DICAST Site Database"
                        site_list[:] = station_numbers
                ds_var = ds.createVariable(variable, "f4", ("recNum", "timesPerDay"))
                date_indices = self.averaged_data[variable].index.date == date
                ds_var[:] = self.averaged_data[variable].loc[date_indices, :].values.T
                ds_var.long_name = variable
                ds_var.units = self.units[variable.split("_")[0]]
                ds.close()
        return