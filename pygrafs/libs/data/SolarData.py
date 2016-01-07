from pvlib.solarposition import get_solarposition
from pvlib.irradiance import extraradiation
from pvlib.location import Location
from pvlib.spa import solar_position_numpy
from ModelGrid import ModelGridSubset
import numpy as np
import pandas as pd


class SolarData(object):
    def __init__(self, times, lon_grid, lat_grid, elevations=None):
        self.times = pd.DatetimeIndex(times)
        self.lon_grid = lon_grid
        self.lat_grid = lat_grid
        self.elevations = elevations

    def solar_position(self, position_variables=["elevation", "zenith", "azimuth", "ETRC"]):
        position_data = {}
        for pos_var in position_variables:
            position_data[pos_var] = np.zeros((self.times.size, self.lon_grid.shape[0], self.lon_grid.shape[1]))
        for (r, c), l in np.ndenumerate(self.lon_grid):
            if self.elevations is not None:
                elev = self.elevations[r, c]
            else:
                elev = 0
            loc = Location(self.lat_grid[r, c], self.lon_grid[r, c], tz="UTC", altitude=elev)
            location_info = get_solarposition(self.times, loc)
            location_info["EXTR"] = extraradiation(self.times, method="pyephem")
            location_info["ETRC"] = location_info["EXTR"] * np.cos(np.radians(location_info["zenith"]))
            location_info.loc[location_info["zenith"] > 90, "ETRC"] = 0
            for pos_var in position_variables:
                position_data[pos_var][:, r, c] = location_info[pos_var].values
        solar_grids = {}
        for pos_var in position_variables:
            solar_grids[pos_var] = ModelGridSubset(pos_var, position_data[pos_var],
                                                   self.times, self.lat_grid, self.lon_grid)
        return solar_grids


def make_solar_position_grid(times, lon_grid, lat_grid, elevations):
    position_variables = ["apparent_zenith", "zenith", "apparent_elevation", "elevation", "azimuth"]
    position_data = np.zeros((len(position_variables), times.size, lon_grid.shape[0], lon_grid.shape[1]), dtype=float)
    pressure = 1013.25
    delta_t = 67.0
    atmos_refract = 0.5667
    temp = 12
    etr = extraradiation(times, method="pyephem").values
    etr = etr.reshape((etr.size, 1, 1))
    unix_times = times.astype(np.int64)
    for (r, c), l in np.ndenumerate(lon_grid):
        position_data[:, :, r, c] = solar_position_numpy(unix_times, lat_grid[r, c], lon_grid[r, c],
                                                         elevations[r, c],
                                                         pressure,
                                                         temp, delta_t,
                                                         atmos_refract, 1)[0:5]
    solar_grids = {}
    for p, pos_var in enumerate(position_variables):
        solar_grids[pos_var] = ModelGridSubset(pos_var, position_data[p],
                                               times, lat_grid, lon_grid, times[0])
    solar_grids["ETRC"] = ModelGridSubset("ETRC", etr * np.cos(np.radians(position_data[0])),
                                          times, lat_grid, lon_grid, times[0])
    solar_grids["ETRC"].data[solar_grids["ETRC"].data < 0] = 0
    return solar_grids

if __name__ == "__main__":
    times = pd.DatetimeIndex(start="2015-06-04", end="2015-06-05 12:00", freq="1H")
    lon_grid, lat_grid = np.meshgrid(np.arange(-110, -90, 0.5), np.arange(30, 40, 0.5))
    elevations = np.zeros(lon_grid.shape)
    pos_data = make_solar_position_grid(times, lon_grid, lat_grid, elevations)
    print pos_data["ETRC"].data.max(), pos_data["ETRC"].data.min()