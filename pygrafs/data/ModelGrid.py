"""
This module loads and performs operations on gridded model output.
"""
from datetime import datetime, timedelta

import numpy as np
from netCDF4 import Dataset, num2date


class ModelGrid(object):
    """
    ModelGrid objects can load model output from netCDF files and perform
    operations on the data. The initialization method opens the file and
    loads latitude and longitude data if it is available.

    :param filename: Full path and name of netCDF file containing model
    information.
    :param x_var: Name of the variable containing x-coordinate information.
    Default "lon".
    :param y_var: Name of the variable containing y-coordinate information.
    Default "lat".
    """

    def __init__(self, filename, x_var="lon", y_var="lat",
                 time_var="forc_time", time_format="grafs_int",delta_t=timedelta(hours=1)):
        self.filename = filename
        self.file_obj = Dataset(self.filename)
        self.x_var = x_var
        self.y_var = y_var
        self.time_var = time_var
        self.time_format = time_format
        self.delta_t = delta_t
        #Initialize empty data dictionary to store information from 1 or more
        #variables.
        self.data = {}
        #Load longitude and latitude grids if available.
        if self.x_var is not None:
            self.x = self.file_obj.variables[x_var][:]
            self.y = self.file_obj.variables[y_var][:]
        else:
            self.x = None
            self.y = None
        if time_format == "grafs_int":
            self.start_date = num2date(self.file_obj.variables[time_var][:],
                                       self.file_obj.variables[time_var].units)
            num_times = len(self.file_obj.dimensions['daily_fcst_times'])
            self.valid_dates = np.array([self.start_date + t * delta_t for t in range(num_times)])
        return

    def load_full(self, variable):
        """
        Loads all available data from a particular variable.

        :param variable: Name of the variable being loaded.
        """
        if variable in self.file_obj.variables.keys():
            var_obj = self.file_obj.variables[variable]
            self.data[variable] = var_obj[:]
            #if hasattr(var_obj, 'scale_factor'):
            #    self.data[variable] = self.data[variable] * var_obj.scale_factor + var_obj.add_offset
        else:
            raise KeyError(variable + " not found")

    def load_subset(self, variable, time_subset, y_subset, x_subset,
                    time_subset_type="index", space_subset_type="index"):
        """
        Load a subset of a variable's data from the file. The subset can be
        specified in time and space with either the array indices ("index") or
        the time stamps, latitudes, and longitudes ("coordinate"). The start
        and end points are inclusive.

        :param variable: Name of the variable being loaded.
        :param time_subset: Tuple of starting and ending times or indices
        :param y_subset: Tuple of the bounds in the y-coordinate dimension.
        :param x_subset: Tuple of the bounds in the x-coordinate dimension.
        :param time_subset_type:
        """
        if variable in self.file_obj.variables.keys():
            if time_subset_type.lower() == "index":
                start_time = time_subset[0]
                end_time = time_subset[1] + 1
            elif time_subset_type.lower() == "coordinate":
                #Compare given datetimes with ones generated from file metadata
                start_time = np.where(self.valid_dates == time_subset[0])[0]
                end_time = np.where(self.valid_dates == time_subset[1])[0] + 1
            else:
                start_time = time_subset[0]
                end_time = time_subset[1] + 1
            if space_subset_type.lower() == "index":
                start_x = x_subset[0]
                end_x = x_subset[1] + 1
                start_y = y_subset[0]
                end_y = y_subset[1] + 1
            elif space_subset_type.lower() == "coordinate":
                start_y, start_x = self.coordinate_to_index(x_subset[0], y_subset[0])
                end_y, end_x = self.coordinate_to_index(x_subset[1], y_subset[1])
            else:
                start_x = 0
                end_x = 0
                start_y = 0
                end_y = 0
            subset_data = self.file_obj.variables[variable][start_time:end_time,
                                                            start_y:end_y, start_x:end_x]
            subset_obj = ModelGridSubset(variable,subset_data,
                                         self.valid_dates[start_time:end_time],
                                         self.y[start_y:end_y,start_x:end_x],
                                         self.x[start_y:end_y,start_x:end_x])
        else:
            raise KeyError(variable + " not found")
        return subset_obj

    def coordinate_to_index(self, x, y):
        """
        Convert x and y coordinates to array indices.

        :param x: x-coordinate of data (float)
        :param y: y-coordinate of data (float)
        :returns: the row and column indices nearest to the given coordinates.
        """
        dist = (self.x - x) ** 2 + (self.y - y) ** 2
        i, j = np.unravel_index(np.argmin(dist), self.x.shape)
        return i, j

    def close(self):
        """
        Closes the connection to the netCDF file.
        """
        self.file_obj.close()
        return


class ModelGridSubset(object):
    """
    Stores data for a subset of a full model grid as well as
    coordinate and time information.

    :param variable: Name of variable being extracted.
    :param data: array of data values
    :param times: array of datetime objects corresponding to data valid times
    :param y: array of y-coordinate values
    :param x: array of x-coordinate values
    """
    def __init__(self, variable, data, times, y, x):
        self.variable = variable
        self.data = data
        self.times = times
        self.y = y
        self.x = x

    def get_point_data(self, time, y_point, x_point, method='nearest'):
        t = self.get_time_index(time)
        i, j = self.coordinate_to_index(x_point, y_point)
        return self.data[t, i, j]

    def get_time_index(self, time):
        return np.argmin(np.abs((self.times - time).total_seconds()))

    def coordinate_to_index(self, x, y):
        """
        Convert x and y coordinates to array indices.

        :param x: x-coordinate of data (float)
        :param y: y-coordinate of data (float)
        :returns: the row and column indices nearest to the given coordinates.
        """
        dist = (self.x - x) ** 2 + (self.y - y) ** 2
        i, j = np.unravel_index(np.argmin(dist), self.x.shape)
        return i, j
