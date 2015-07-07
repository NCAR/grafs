from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd


def main():
    var = "av_dswrf_sfc"
    obs = ObsSite("/d2/dgagne/grafs/pygrafs/test/test_data/int_obs.20141215.nc")
    obs.load_data(var)
    print obs.data[var]
    return


class ObsSite(object):
    def __init__(self, filename,
                 file_format='nc',
                 meta_file="/d2/dgagne/static_data/site_list/int_obs_sites_solar.asc",
                 meta_delimiter=';',
                 meta_index_col="stationNumber",
                 time_var="time_nominal"):
        self.filename = filename
        self.file_format = file_format
        self.meta_file = meta_file
        self.meta_delimiter = meta_delimiter
        self.meta_index_col = meta_index_col
        self.meta_data = self.load_meta_file()
        self.time_var = time_var
        self.file_obj = Dataset(self.filename)
        self.times = num2date(self.file_obj.variables[time_var][:],
                              self.file_obj.variables[time_var].units)
        self.data = {}
        self.station_data = None
        return

    def load_meta_file(self):
        """
        Loads station information from binary file.
        
        :return: pandas DataFrame containing locations and names for the
            available observation sites
        """
        meta_data = pd.read_csv(self.meta_file,
                                sep=self.meta_delimiter,
                                index_col=self.meta_index_col)
        return meta_data

    def load_data(self, variable):
        """
        Read observations from file.

        :param variable: Name of the variable being loaded from the file.
        """
        # Load variable data from file
        all_data = self.file_obj.variables[variable][:]
        # Determine which rows in the data have valid observations
        valid_rows = np.unique(np.nonzero(all_data < all_data.max())[0])
        # Mask out data values that are invalid
        data = np.ma.array(all_data[valid_rows],
                           mask=all_data[valid_rows] == all_data.max())
        # Get valid stations
        stations = self.file_obj.variables['site_list'][valid_rows]
        station_codes = self.meta_data.loc[stations, "solar_code"].values.astype(int)
        self.station_data = self.meta_data.loc[stations]
        flat_dict = {'station': [], 'valid_date': [], variable: []}

        # Loop through all data values and add metadata and data values to data structure
        for (s, d), v in np.ndenumerate(data):
            if station_codes[s] == 8:
                flat_dict['station'].append(stations[s])
                flat_dict['valid_date'].append(self.times[d])
                if v > 1e30:
                    flat_dict[variable].append(np.nan)
                else:
                    flat_dict[variable].append(v)
        # Convert data to data frame.
        flat_data = pd.DataFrame(flat_dict)
        # Remove rows with nan values
        flat_data = flat_data.dropna()
        # Join latitude and longitude information to station observations
        self.data[variable] = pd.merge(flat_data, self.station_data[["lat", "lon"]],
                                       left_on="station", right_index=True, how="left")
        return

    def filter_by_location(self, variable, lon_bounds, lat_bounds):
        """
        Remove observations from data that fall outside the specified latitude and longitude bounds.

        :param lon_bounds: tuple of lower and upper bounds of longitudes
        :param lat_bounds: tuple of lower and upper bounds of latitudes
        """
        self.data[variable] = self.data[variable].ix[(self.data[variable]['lon'] >= lon_bounds[0])
                                                     & (self.data[variable]['lon'] <= lon_bounds[1])
                                                     & (self.data[variable]['lat'] >= lat_bounds[0])
                                                     & (self.data[variable]['lat'] <= lat_bounds[1])]
        self.data[variable].reset_index(drop=True, inplace=True)
        self.station_data = self.station_data.ix[(self.station_data['lon'] >= lon_bounds[0])
                                                 & (self.station_data['lon'] <= lon_bounds[1])
                                                 & (self.station_data['lat'] >= lat_bounds[0])
                                                 & (self.station_data['lat'] <= lat_bounds[1])]

    def close(self):
        """
        Close netCDF file object.

        :return:
        """
        self.file_obj.close()


if __name__ == "__main__":
    main()