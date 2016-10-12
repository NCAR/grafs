__author__ = 'David John Gagne'

import gdal, osr
import numpy as np
import pyproj
from netCDF4 import Dataset


def main():
    path = "/d2/dgagne/USGS_Data/GTOPO30 HYDRO 1K/gt30h1kna/"
    lon_lim = (-125, -55)
    lat_lim = (20, 50)
    vars = ['slope', 'asp']
    data_dict = {}
    print('dem')
    data, lon_grid, lat_grid = load_data(path + "na_dem.bil")
    sub_lon, sub_lat, sub_data = get_grid_subset(lon_lim, lat_lim, lon_grid, lat_grid, data)
    data_dict['dem'] = sub_data
    for var in vars:
        print(var)
        filename = path + "na_{0}.bil".format(var)
        data = load_data(filename, projection=False)
        sub_lon, sub_lat, sub_data = get_grid_subset(lon_lim, lat_lim, lon_grid, lat_grid, data)
        data_dict[var] = sub_data
    out_filename = "/d2/dgagne/GTOPO30_HYDRO_1K_terrain.nc"
    print("Output to netCDF")
    data_to_netcdf(out_filename, data_dict, sub_lon, sub_lat)
    return


def load_data(filename, projection=True):
    img = gdal.Open(filename)
    band = img.GetRasterBand(1)
    data = band.ReadAsArray()
    max_val = band.GetMaximum()
    data[data > max_val] = 0
    if 'dem' not in filename:
        data = data / 100.0
    if projection:
        geo_trans = img.GetGeoTransform()
        xs = np.arange(geo_trans[0], geo_trans[0] + geo_trans[1] * data.shape[1], geo_trans[1])
        ys = np.arange(geo_trans[3], geo_trans[3] + geo_trans[5] * data.shape[0], geo_trans[5])
        x_grid, y_grid = np.meshgrid(xs, ys)
        srs = osr.SpatialReference(wkt=img.GetProjection())
        projection = pyproj.Proj(srs.ExportToProj4())
        lon_grid, lat_grid = projection(x_grid, y_grid, inverse=True)
        return data, lon_grid, lat_grid
    else:
        return data


def get_grid_subset(lon_lim, lat_lim, lon_grid, lat_grid, data):
    ll_bound = coord_to_index(lon_lim[0], lat_lim[0], lon_grid, lat_grid)
    ur_bound = coord_to_index(lon_lim[1], lat_lim[1], lon_grid, lat_grid)
    print(ll_bound, ur_bound)
    i_slice = slice(np.minimum(ll_bound[0], ur_bound[0]), np.maximum(ll_bound[0], ur_bound[0]))
    j_slice = slice(np.minimum(ll_bound[1], ur_bound[1]), np.maximum(ll_bound[1], ur_bound[1]))
    sub_lon = lon_grid[i_slice, j_slice]
    sub_lat = lat_grid[i_slice, j_slice]
    sub_data = data[i_slice, j_slice]
    return sub_lon, sub_lat, sub_data


def coord_to_index(x, y, x_g, y_g):
    dist = (x_g - x) ** 2 + (y_g - y) ** 2
    i, j = np.unravel_index(np.argmin(dist), x_g.shape)
    return i, j


def data_to_netcdf(out_filename, data_dict, lon_grid, lat_grid):
    long_names = dict(dem="Digital Elevation model", asp="Terrain aspect angle", slope="Slope angle of terrain")
    out_obj = Dataset(out_filename, 'w')
    out_obj.createDimension('y', lon_grid.shape[0])
    out_obj.createDimension('x', lon_grid.shape[1])
    lon_var = out_obj.createVariable('lon', 'f4', ('y', 'x'))
    lon_var[:] = lon_grid
    lon_var.long_name = "longitude"
    lat_var = out_obj.createVariable('lat', 'f4', ('y', 'x'))
    lat_var[:] = lat_grid
    lat_var.long_name = "latitude"
    for var in data_dict.keys():
        data_var = out_obj.createVariable(var, 'f4', ('y', 'x'))
        data_var[:] = data_dict[var]
        data_var.long_name = long_names[var]
        if var in ['dem']:
            data_var.units = 'm'
        if var in ['aspect', 'slope']:
            data_var.units = 'degrees'
    out_obj.close()
    return
if __name__ == "__main__":
    main()
