from netCDF4 import Dataset
from scipy.interpolate import NearestNDInterpolator
import numpy as np


def main():
    from ModelGrid import ModelGrid
    import matplotlib.pyplot as plt
    model_grid = ModelGrid("../../test/test_data/int_fcst_grid.20141102.11.nc")
    t_range = (0, 2)
    y_range = (26.0, 43.0)
    x_range = (-125.0, -93.0)
    print "loading subset"
    subset_data = model_grid.load_subset("av_dswrf_sfc", t_range, y_range, x_range, space_subset_type="coordinate")
    land_grid = LandGrid("/d2/dgagne/GTOPO30_HYDRO_1K_terrain.nc")
    print "interpolating"
    interp_data = land_grid.interpolate_grid(subset_data.x, subset_data.y)
    print "plotting"
    plt.contourf(subset_data.x, subset_data.y, interp_data)
    plt.colorbar()
    plt.show()


class LandGrid(object):
    """
    LandGrid manages data from a netCDF file containing gridded land information
    such as elevation or land cover.

    :param filename: Name of the file being loaded.
    :param lon_var: Name of the variable containing longitude information
    :param lat_var: Name of the variable containing latitude information
    :param data_vars: List of variables from file being loaded
    """
    def __init__(self, filename, lon_var='lon', lat_var='lat', data_vars=['dem']):
        self.filename = filename
        self.lon_var = lon_var
        self.lat_var = lat_var
        self.data_vars = data_vars
        try:
            land_obj = Dataset(self.filename)
            self.lon = land_obj.variables[self.lon_var][:]
            self.lat = land_obj.variables[self.lat_var][:]
            self.data = {}
            for data_var in data_vars:
                self.data[data_var] = land_obj.variables[data_var[:]]
        finally:
            land_obj.close()

    def interpolate_grid(self, var, lon_grid, lat_grid):
        """
        Interpolate native data to given lat-lon grid using nearest neighbor interpolation.

        :param lon_grid: 2-d array of longitude values
        :param lat_grid: 2-d array of latitude values
        :return: interpolated values as a 2-d array with the same shape as the longitude grid
        """
        interper = NearestNDInterpolator(np.vstack((self.lon.ravel(), self.lat.ravel())).T, self.data[var].ravel())
        interp_values = interper(np.vstack((lon_grid.ravel(), lat_grid.ravel())).T)
        return interp_values.reshape(lon_grid.shape)

if __name__ == "__main__":
    main()