import numpy as np
import pandas as pd
from pygrafs.libs.data.ObsSite import ObsSeries
from pygrafs.libs.model.gridding import cressman


class Persistence(object):
    def __init__(self, variable, offset_hours, obs_path, lon_bounds, lat_bounds):
        self.variable = variable
        self.offset_hours = offset_hours
        self.lon_bounds = lon_bounds
        self.lat_bounds = lat_bounds
        self.obs_path = obs_path

    def make_predictions(self, eval_data, test_data, x_name="lon", y_name="lat"):
        unique_dates = np.unique(pd.DatetimeIndex(test_data["valid_date"]))
        predictions = test_data.loc[:, ["valid_date", "station", x_name, y_name]]
        pred_dates = pd.DatetimeIndex(predictions["valid_date"])
        predictions.loc[:, "Persistence"] = 0.5
        train_sites = np.unique(eval_data["station"].values)
        for unique_date in unique_dates:
            print(unique_date)
            persist_date = unique_date - pd.Timedelta(24, unit="h")
            obs_series = ObsSeries(self.variable, np.array([persist_date]), self.lon_bounds, self.lat_bounds, "/d2/dgagne/mesonet_nc_2/mesonet")
            obs_series.load_data()
            if obs_series.data.shape[0] > 0 and np.count_nonzero(pred_dates == unique_date) > 1:
                predictions.loc[pred_dates == unique_date, "Persistence"] = cressman(obs_series.data.loc[np.in1d(obs_series.data["station"].values, train_sites), 
                                                                                                                 [x_name, y_name, self.variable]],
                                                                                     predictions.loc[pred_dates == unique_date, [x_name, y_name]], 
                                                                                     y_name,
                                                                                     x_name)
        return predictions

