from netCDF4 import Dataset,num2date
import pandas as pd

class ObsSite(object):
    def __init__(self, filename,
                 file_format='nc',
                 meta_file="/d2/dicast/nt/static_data/site_list/int_obs_sites.asc",
                 meta_delimiter='csv',
                 time_var="time_nominal"):
        self.filename = filename
        self.file_format = file_format
        self.meta_file = meta_file
        self.meta_delimiter = meta_delimiter
        self.time_var = time_var
        self.file_obj = Dataset(self.filename)
        self.times = num2date(self.file_obj.variables[time_var][:],
                              self.file_obj.variables[time_var].units)
        self.data = {}
        return

    def load_meta_file(self):
        pd.read_csv(self.meta_file, delimiter=self.meta_delimiter,header=None)
        return

    def load_data(self, variable):
        self.data[variable] = self.file_obj.variables[variable][:]
        return

    def close(self):
        self.file_obj.close()