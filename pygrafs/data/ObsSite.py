from netCDF4 import Dataset,num2date
import numpy as np
import pandas as pd



def main():
    var = "av_dswrf_sfc"
    obs = ObsSite("/d2/dgagne/grafs/pygrafs/test/test_data/int_obs.20141215.nc")
    obs.load_data(var)
    obs.calc_clearsky(var)
    print obs.data[var]
    return


class ObsSite(object):
    def __init__(self, filename,
                 file_format='nc',
                 meta_file="/d2/dgagne/static_data/site_list/int_obs_sites.asc",
                 meta_delimiter=';',
                 time_var="time_nominal"):
        self.filename = filename
        self.file_format = file_format
        self.meta_file = meta_file
        self.meta_delimiter = meta_delimiter
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
        col_names = ["index","synop","icao","lat","lon","elev","plot","longname","state","country"]
        meta_data = pd.read_csv(self.meta_file,
                    sep=self.meta_delimiter,
                    header=None,
                    names=col_names,
                    index_col="index")
        return meta_data

    def load_data(self, variable):
        """
        Read observations from file.
        """
        all_data = self.file_obj.variables[variable][:]
        valid_rows = np.unique(np.nonzero(all_data < all_data.max())[0])
        print valid_rows.shape, all_data.max(), all_data.min()
        data = np.ma.array(all_data[valid_rows],
                                          mask=all_data[valid_rows] == all_data.max())
        stations = self.file_obj.variables['site_list'][valid_rows]
        self.station_data = self.meta_data.loc[stations]
        flat_dict = {'station': [], 'date': [], variable: []}
        for (s, d), v in np.ndenumerate(data):
            flat_dict['station'].append(stations[s])
            flat_dict['date'].append(self.times[d])
            if v > 1e30:
                flat_dict[variable].append(np.nan)
            else:
                flat_dict[variable].append(v)
        flat_data = pd.DataFrame(flat_dict)
        flat_data = flat_data.dropna()
        self.data[variable] = pd.merge(flat_data, self.station_data[["lat","lon"]], left_on="station", right_index=True, how="left")
        print "Obs data shapes: ", flat_data.shape, self.data[variable].shape
        return


    def close(self):
        self.file_obj.close()


if __name__ == "__main__":
    main()