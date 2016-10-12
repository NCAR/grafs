__author__ = 'David John Gagne'
from netCDF4 import Dataset, chartostring
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import ListedColormap


def main():
    station_file = "/d2/dgagne/static_data/site_list/int_obs_sites.asc"
    madis_filename = "/d2/ldm/data/dec_data/obs/madis/20150531/madis.20150531.1500.nc"
    provider_table, obs_table, station_table = get_provider_info(madis_filename)
    grafs_stations = load_meta_file(station_file)
    print(provider_table.loc[provider_table['solar_code'] > 0])
    combined = pd.merge(grafs_stations,
                        station_table[['StationID', 'Provider', 'solar_code', 'solar_description']],
                        left_on="icao", right_on="StationID")
    unique_codes = combined['solar_description'].unique()
    for code in unique_codes:
        print(code, combined.loc[combined['solar_description'] == code].shape[0])
    grafs_providers = combined['Provider'].unique()
    for provider in grafs_providers:
        provider_table.loc[provider, 'station_count'] = np.count_nonzero(combined['Provider'] == provider)
    print(provider_table.loc[grafs_providers])
    print(combined.columns)
    combined.to_csv("/d2/dgagne/static_data/site_list/int_obs_sites_solar.asc", sep=";", index=False)
    cm = ['red', 'orange', 'green', 'blue', 'purple']
    plt.figure(figsize=(10, 7))
    bmap = Basemap(projection='cyl', resolution='i', llcrnrlon=-125, llcrnrlat=25, urcrnrlon=-90, urcrnrlat=45)
    bmap.drawstates()
    bmap.drawcoastlines()
    bmap.drawcountries()
    for c, code in enumerate(unique_codes):
        idxs = combined['solar_description']==code
        plt.scatter(combined['lon'][idxs], combined['lat'][idxs],20, cm[c], edgecolors='', label=code.split("=")[-1].strip())
    plt.legend(loc=0, fontsize=10)
    plt.title("GRAFS Site Solar Radiation Averaging Interval")
    plt.savefig("/d2/dgagne/grafs_figures/averaging_interval_map.png", dpi=200, bbox_inches='tight')
    plt.show()
    return


def get_provider_info(madis_filename):
    madis_data = Dataset(madis_filename)
    data_providers = chartostring(madis_data.variables['dataProvider'][:])
    station_id = chartostring(madis_data.variables['stationId'][:])
    solar_radiation = madis_data.variables['solarRadiation'][:]
    provider_ids = np.array([x.strip() for x in chartostring(madis_data.variables["namePST"][:])])
    obs_times = pd.DatetimeIndex(pd.to_datetime(madis_data.variables["observationTime"][:], unit='s'))
    lat = madis_data.variables['latitude'][:]
    lon = madis_data.variables['longitude'][:]
    good_ids = np.where(provider_ids != "")[0]
    provider_table = pd.DataFrame(index=provider_ids[good_ids])
    provider_table['solar_code'] = madis_data.variables['code2PST'][good_ids]
    provider_table['solar_description'] = [getattr(madis_data.variables['code2PST'],
                                                   "value{0:d}".format(x)) for x in provider_table['solar_code']]

    madis_data.close()
    print(data_providers)
    obs_table = pd.DataFrame({"Provider": data_providers,
                              "StationID": station_id,
                              "SolarRadiation": solar_radiation,
                              "Lat": lat,
                              "Lon": lon,
                              "Time": obs_times})
    obs_table.drop_duplicates(subset=["Provider", "StationID"], inplace=True)
    obs_table.reset_index(inplace=True, drop=True)
    station_counts = np.zeros(len(good_ids))
    for p, provider in enumerate(provider_ids[good_ids]):
        station_counts[p] = np.count_nonzero(obs_table["Provider"] == provider)
    provider_table['station_count'] = station_counts
    station_table = pd.merge(obs_table.loc[:, ['Provider', 'StationID', 'Lat', 'Lon']],
                             provider_table, left_on="Provider", right_index=True, how="left")
    print(station_table.shape, obs_table.shape)
    return provider_table, obs_table, station_table


def load_meta_file(meta_file, delimiter=";"):
        """
        Loads station information from binary file.

        :return: pandas DataFrame containing locations and names for the
            available observation sites
        """
        col_names = ["stationNumber", "synop", "icao", "lat", "lon", "elev", "plot", "longname", "state", "country"]
        meta_data = pd.read_csv(meta_file,
                                sep=delimiter,
                                header=None,
                                names=col_names)
        return meta_data


if __name__ == "__main__":
    main()
