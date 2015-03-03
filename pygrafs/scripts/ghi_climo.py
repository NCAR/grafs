import os
from netCDF4 import Dataset, date2num
import sys

from scipy.io import loadmat
import numpy as np
import pandas as pd
from pvlib.clearsky import clearsky_ineichen
from pvlib.location import Location

from pygrafs.libs.data.ModelGrid import ModelGrid


def main():
    grid_file = "/d2/dgagne/grafs/pygrafs/test/test_data/int_fcst_grid.20141102.11.nc"
    linke_mat_file = "/d2/dgagne/PVLIB_Python/pvlib/data/LinkeTurbidities.mat"
    linke_nc_file = "/d2/dgagne/grafs/pygrafs/data/LinkeTurbidities.nc"
    elevation_file = "/d2/dgagne/elevation/ETOPO2v2g_f4.nc"
    ghi_file = "/d2/dgagne/ghi_climo.nc"
    start_time = "2012-01-01T00:00Z"
    end_time = "2012-12-31T23:00Z"
    if not os.access(linke_nc_file, os.R_OK):
        convert_linke_turbidity(linke_mat_file, linke_nc_file)
    generate_clear_sky_climo(grid_file, linke_nc_file, elevation_file, ghi_file, start_time, end_time)
    return


def generate_clear_sky_climo(grid_file, linke_file, elevation_file, out_file, start_time, end_time, time_units='seconds since 1970-01-01 00:00:00'):
    mg = ModelGrid(grid_file)
    lon_flat = np.arange(np.round(mg.x.min()), np.round(mg.x.max()) + 0.25, 0.25)
    lat_flat = np.arange(np.round(mg.y.min()), np.round(mg.y.max()) + 0.25, 0.25)
    mg.close()
    lons, lats = np.meshgrid(lon_flat, lat_flat)
    elev = Elevation(elevation_file)
    times = pd.date_range(start_time, end_time, freq='H', tz='UTC')
    linke_obj = LinkeTurbidity(linke_file)
    out_file_obj = Dataset(out_file, 'w')
    try:
        out_file_obj.createDimension('y', lons.shape[0])
        out_file_obj.createDimension('x', lons.shape[1])
        out_file_obj.createDimension('time', times.size)
        lon_var =  out_file_obj.createVariable('lon', 'f4', ('y', 'x'))
        lat_var = out_file_obj.createVariable('lat', 'f4', ('y', 'x'))
        time_var = out_file_obj.createVariable('time', 'f8', ('time',))
        day_var = out_file_obj.createVariable('dayofyear', 'i2', ('time',))
        day_var.long_name = 'day of year'
        hour_var = out_file_obj.createVariable('hour','i2',('time',))
        hour_var.long_name = 'valid hour'
        print "Creating GHI Var"
        ghi_var = out_file_obj.createVariable('GHI', 'f4', ('y', 'x', 'time'))
        ghi_var.units = 'W m-2'
        ghi_var.long_name = "Global Horizontal Irradiance"
        lon_var[:] = lons
        lat_var[:] = lats
        time_var[:] = date2num(times.to_pydatetime(), time_units)
        day_var[:] = times.dayofyear
        hour_var[:] = times.hour
        flat_index = 0
        ghi_arr = np.empty((lons.shape[0],lons.shape[1],times.size), dtype=np.float32)
        print ghi_arr.shape
        while flat_index < lons.size:
            row, col = np.unravel_index(flat_index,lons.shape)
            proc_name = "{0:03d},{1:03d}".format(row, col)
            sys.stdout.write("\r" + proc_name + " started")
            sys.stdout.flush()
            elevation = elev.get_elevation(lons[row, col], lats[row, col])
            args = (lons[row, col], lats[row, col], elevation, start_time, end_time, linke_obj)
            ghi_arr[row,col] = apply(calc_point_clear_sky, args)
            flat_index += 1
        out_file_obj.variables['GHI'][:] = ghi_arr
    finally:
        out_file_obj.close()
    return


def calc_point_clear_sky(lon, lat, elevation, start_time, end_time, linke_obj):
    turbidity = linke_obj.get_turbidity_values(lon, lat, start_time, end_time)
    times = pd.date_range(start_time, end_time, freq='H', tz='UTC')
    location = Location(lat, lon, tz='UTC', altitude=elevation)
    radiation = clearsky_ineichen(times, location, turbidity['data'].values)
    return radiation['ClearSkyGHI'].values


class Elevation(object):
    def __init__(self, elevation_file):
        elev_obj = Dataset(elevation_file)
        self.lon = elev_obj.variables['x'][:]
        self.lat = elev_obj.variables['y'][:]
        self.elevation = elev_obj.variables['z'][:]
        self.elevation[self.elevation < 0] = 0
        elev_obj.close()

    def get_nearest_point(self, point_lon, point_lat):
        i = np.argmin(np.abs(self.lat - point_lat))
        j = np.argmin(np.abs(self.lon - point_lon))
        return i, j

    def get_elevation(self, lon, lat):
        return self.elevation[self.get_nearest_point(lon, lat)]


class LinkeTurbidity(object):
    def __init__(self, filename):
        linke_file = Dataset(filename)
        self.lon = linke_file.variables['lon'][:]
        self.lat = linke_file.variables['lat'][:]
        self.month = linke_file.variables['month'][:]
        self.data = linke_file.variables['LinkeTurbidity'][:]
        linke_file.close()

    def get_nearest_point(self, point_lon, point_lat):
        i = np.argmin(np.abs(self.lat - point_lat))
        j = np.argmin(np.abs(self.lon - point_lon))
        return i, j

    def get_turbidity_values(self, point_lon, point_lat, start_time, end_time, freq="H"):
        time_series = pd.date_range(np.datetime64(start_time) - np.timedelta64(30,'D'),
                      np.datetime64(end_time) + np.timedelta64(30,'D'),
                      freq=freq, tz='UTC')
        i, j = self.get_nearest_point(point_lon, point_lat)
        monthly_turbidity = pd.DataFrame(dict(data=np.tile(self.data[i, j],3)),
                                         index=pd.DatetimeIndex(pd.date_range('2011-01-31',
                                                                              '2013-12-31',
                                                                              freq='M',
                                                                              tz='UTC').shift(-15,'D')))
        all_turbidity = pd.DataFrame(index=pd.DatetimeIndex(time_series))
        all_turbidity = pd.merge(all_turbidity, monthly_turbidity, how='left',left_index=True, right_index=True)
        all_turbidity['data'] = all_turbidity['data'].interpolate('cubic')
        st = all_turbidity.index.searchsorted(np.datetime64(start_time))
        et = all_turbidity.index.searchsorted(np.datetime64(end_time))
        all_turbidity = all_turbidity.ix[st:et+1]
        return all_turbidity

def convert_linke_turbidity(linke_mat_file, linke_nc_file):
    """
    Read linke turbidity mat file and convert it to a netCDF file.

    :param linke_mat_file: name of mat file containing linke turbidities
    :param linke_nc_file: name of netCDF file in which Linke turbidities are written
    :return:
    """
    linke_mat = loadmat(linke_mat_file)
    linke_data = linke_mat['LinkeTurbidity']
    lons = np.linspace(-180, 180, linke_data.shape[1])
    lats = np.linspace(90, -90, linke_data.shape[0])
    out_file = Dataset(linke_nc_file, 'w')
    out_file.title = "World Linke Turbidity Monthly Climatology"
    out_file.createDimension('lat',lats.size)
    out_file.createDimension('lon',lons.size)
    out_file.createDimension('month',linke_data.shape[2])
    out_file.createVariable('LinkeTurbidity','i4',dimensions=('lat','lon','month'))
    out_file.variables['LinkeTurbidity'][:] = linke_data
    out_file.variables['LinkeTurbidity'].scale_factor = 0.05
    out_file.variables['LinkeTurbidity'].add_offset = 0
    out_file.variables['LinkeTurbidity'].units = ''
    out_file.variables['LinkeTurbidity'].long_name = 'Linke Turbidity Factor'
    out_file.createVariable('lat','f4',('lat'))
    out_file.createVariable('lon','f4',('lon'))
    out_file.createVariable('month','i4',('month'))
    out_file.variables['lat'][:] = lats
    out_file.variables['lon'][:] = lons
    out_file.variables['month'][:] = np.arange(1,13)
    out_file.close()
    return

if __name__ == "__main__":
    main()