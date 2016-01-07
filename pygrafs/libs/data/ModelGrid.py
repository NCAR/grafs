"""
This module loads and performs operations on gridded model output.
"""
from datetime import timedelta

import numpy as np
from netCDF4 import Dataset, num2date
from scipy.ndimage.filters import convolve, maximum_filter, minimum_filter, \
    median_filter, correlate, gaussian_gradient_magnitude


class ModelGrid(object):
    """
    ModelGrid objects can load model output from netCDF files and perform
    operations on the data. The initialization method opens the file and
    loads latitude and longitude data if it is available.

    :param filename: Full path and name of netCDF file containing model information.
    :param x_var: Name of the variable containing x-coordinate information. Default "lon".
    :param y_var: Name of the variable containing y-coordinate information. Default "lat".
    """

    def __init__(self, filename, x_var="lon", y_var="lat",
                 time_var="forc_time", time_format="grafs_int", delta_t=timedelta(hours=1)):
        self.filename = filename
        self.file_obj = Dataset(self.filename)
        self.x_var = x_var
        self.y_var = y_var
        self.time_var = time_var
        self.time_format = time_format
        self.delta_t = delta_t
        # Initialize empty data dictionary to store information from 1 or more
        # variables.
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

            time_dim = 'fcst_times' if 'fcst_times' in self.file_obj.dimensions.keys() else 'daily_fcst_times'
            num_times = len(self.file_obj.dimensions[time_dim])
            self.all_dates = np.array([self.start_date + t * delta_t for t in range(num_times)])
            self.valid_dates = None
        return

    def load_full(self, variable):
        """
        Loads all available data from a particular variable.

        :param variable: Name of the variable being loaded.
        """
        if variable in self.file_obj.variables.keys():
            var_obj = self.file_obj.variables[variable]
            self.data[variable] = var_obj[:]
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
                # Compare given datetimes with ones generated from file metadata
                start_time = np.where(self.all_dates == time_subset[0])[0]
                end_time = np.where(self.all_dates == time_subset[1])[0] + 1
            elif time_subset_type.lower() == "hours":
                start_time = np.where(self.all_dates == self.all_dates[0] + timedelta(hours=time_subset[0]))[0]
                end_time = np.where(self.all_dates == self.all_dates[0] + timedelta(hours=time_subset[1]))[0] + 1
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
                                                            start_y:end_y,
                                                            start_x:end_x]

            subset_data[subset_data < -30000] = np.nan
            subset_data[subset_data > 100000] = np.nan
            subset_obj = ModelGridSubset(variable, subset_data,
                                         self.all_dates[start_time:end_time],
                                         self.y[start_y:end_y, start_x:end_x],
                                         self.x[start_y:end_y, start_x:end_x],
                                         self.all_dates[0])
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
    def __init__(self, variable, data, times, y, x, init_time):
        self.variable = variable
        self.data = data
        self.times = times
        self.y = y
        self.x = x
        self.valid_time_indices = self.get_valid_time_indices()
        self.valid_times = self.times[self.valid_time_indices]
        self.init_time = init_time

    def get_point_data(self, time, y_point, x_point):
        """
        Extract the value of a grid point at a particular time.

        :param time: datetime object corresponding to time of interest
        :param y_point: latitude of point
        :param x_point: longitude of point
        :return: value of data at point
        """
        t = self.get_time_index(time)
        i, j = self.coordinate_to_index(x_point, y_point)
        if ~(np.isnan(i) | np.isnan(j)):
            value = self.data[t, i, j]
        else:
            value = np.nan
        return value

    def get_neighbor_grid_stats(self, time_index, neighbor_radius=1, stats=('mean', 'min', 'max')):
        """
        Calculate grid point neighborhood statistics for every grid point at once.

        :param time_index: Index of the time slice of interest
        :param neighbor_radius: Radius of neighborhood box in grid points
        :param stats: List of statistics being calculated. Mean, min, and max are currently supported.
        :return: Array of statistics with the first dimension corresponding to the order of stats.
        """
        data = self.data[time_index]
        window_size = 1 + 2 * neighbor_radius
        window = np.ones((window_size, window_size), dtype=int)
        stat_arrays = np.zeros((len(stats), data.shape[0], data.shape[1]))
        for s, stat in enumerate(stats):
            if stat == 'mean':
                stat_arrays[s] = convolve(data, window, mode='reflect') / float(window.size)
            elif stat == 'min':
                stat_arrays[s] = minimum_filter(data, footprint=window, mode='reflect')
            elif stat == 'max':
                stat_arrays[s] = maximum_filter(data, footprint=window, mode='reflect')
            elif stat == 'correlate':
                stat_arrays[s] = correlate(data, window, mode='reflect')
            elif stat == 'median':
                stat_arrays[s] = median_filter(data, footprint=window, mode='reflect')
            elif stat == 'gradient':
                stat_arrays[s] = gaussian_gradient_magnitude(data, sigma=neighbor_radius, mode='reflect')
        return stat_arrays

    def get_neighbor_stats(self, time, y_point, x_point, indices=False, neighbor_radius=1,
                           stats=('mean', 'min', 'max', 'gradient')):
        """
        Calculate statistics about the immediate neighborhood of a grid point.

        :param time: datetime object or index corresponding to time of interest
        :param y_point: latitude or row index of point
        :param x_point: longitude or column index of point
        :param indices: If false, convert input coordinates to indices. If true, input coordinates are array indices
        :param neighbor_radius: number of grid points to in each direction to include in neighborhood
        :param stats: list of statistics to be calculated at that point
        :return: numpy.array of statistics.
        """
        if indices:
            t = time
            i = y_point
            j = x_point
        else:
            t = self.get_time_index(time)
            i, j = self.coordinate_to_index(x_point, y_point)
        if ~(np.isnan(i) | np.isnan(j)):
            values = self.data[t,
                               np.maximum(i - neighbor_radius, 0):
                               np.minimum(i + neighbor_radius + 1, self.data.shape[1]),
                               np.maximum(j - neighbor_radius, 0):
                               np.minimum(j + neighbor_radius + 1, self.data.shape[2])]
            stat_values = np.zeros(len(stats))
            for s, stat in enumerate(stats):
                if stat in ['mean', 'min', 'max', 'std', 'var']:
                    stat_values[s] = getattr(values, stat)()
                elif stat in ['median']:
                    stat_values[s] = np.median(stat)
                elif stat in ['gradient']:
                    grad_x, grad_y = np.gradient(values)
                    stat_values[s] = np.sqrt(grad_x[neighbor_radius, neighbor_radius] ** 2
                                             + grad_y[neighbor_radius, neighbor_radius] ** 2)
                else:
                    stat_values[s] = np.nan
        else:
            stat_values = np.ones(len(stats)) * np.nan
        return stat_values

    def get_time_index(self, time):
        """
        Calculates the array index of the timestamp closet to the input time.

        :param time: a datetime object
        :return: array index of closest timestamp
        """
        return np.argmin(np.abs((self.times - time).total_seconds()))
    
    def get_unique_dates(self):
        """
        Extract the individual dates covered by the model run

        :return: array of date objects
        """
        return np.unique(np.array([t.date() for t in self.valid_times]))

    def get_valid_time_indices(self):
        """
        Get the array indices of times that have valid data

        :return:
        """
        if hasattr(self.data, "mask"):
            valid_indices = np.where(np.any(np.any(~self.data.mask, axis=2), axis=1))[0]
        else:
            valid_indices = np.where(np.any(np.any(self.data < 9e30, axis=2), axis=1))[0]
        return np.array(valid_indices, dtype=int)

    def coordinate_to_index(self, x, y):
        """
        Convert x and y coordinates to array indices.

        :param x: x-coordinate of data (float)
        :param y: y-coordinate of data (float)
        :returns: the row and column indices nearest to the given coordinates.
        """
        if (x >= self.x.min()) and (x <= self.x.max()) and (y >= self.y.min()) and (y <= self.y.max()):
            dist = (self.x - x) ** 2 + (self.y - y) ** 2
            i, j = np.unravel_index(np.argmin(dist), self.x.shape)
        else:
            i = np.nan
            j = np.nan
        return i, j
