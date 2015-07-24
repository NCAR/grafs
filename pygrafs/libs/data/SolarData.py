from pvlib.solarposition import get_solarposition
from pvlib.irradiance import extraradiation
from pvlib.location import Location
from pvlib.clearsky import haurwitz
import numpy as np


class SolarData(object):
    def __init__(self, times, lon_grid, lat_grid, elevations=None):
        self.times = times
        self.lon_grid = lon_grid
        self.lat_grid = lat_grid
        self.elevations = elevations

    def solar_position(self, position_variables=["elevation", "zenith", "azimuth", "etr_cos"]):
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
            location_info["etr"] = extraradiation(self.times, method="pyephem")
            location_info["etr_cos"] = location_info["etr"] * np.cos(np.radians(location_info["zenith"]))
            location_info.loc[location_info["zenith"] > 90, "etr_cos"] = 0
            for pos_var in position_variables:
                position_data[pos_var][:, r, c] = location_info[pos_var].values
        return position_data