from netCDF4 import Dataset


class LandGrid(object):
    """
    LandGrid manages data from a netCDF file containing gridded land information
    such as elevation or land cover.

    :param filename: Name of the file being loaded.
    :param lon_var: Name of the variable containing longitude information
    :param lat_var: Name of the variable containing latitude information
    """
    def __init__(self, filename, lon_var='x', lat_var='y', data_var='z'):
        self.filename = filename
        self.lon_var = lon_var
        self.lat_var = lat_var
        self.data_var = data_var
        try:
            land_obj = Dataset(self.filename)
            self.lon = land_obj.variables[self.lon_var][:]
            self.lat = land_obj.variables[self.lat_var][:]
            self.data = land_obj.variables[self.data_var][:]
        finally:
            land_obj.close()



