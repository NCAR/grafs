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
        unique_dates = pd.DatetimeIndex(np.unique(test_data["valid_date"]))
        predictions = test_data.loc[:, ["valid_date", "station", x_name, y_name]]
        predictions.loc[:, "Persistence"] = 0.5
        for unique_date in unique_dates:
            persist_date = unique_date - pd.Timedelta(24, unit="h")
            obs_series = ObsSeries(self.variable, np.array([persist_date]), self.lon_bounds, self.lat_bounds)
            obs_series.load_data()
            train_sites = eval_data.loc[eval_data["valid_date"] == unique_date, "station"]
            predictions.loc[predictions["valid_date"] == unique_date, "Persistence"] = cressman(obs_series.data.loc[np.in1d(obs_series.data["station"], train_sites)],
                                                                                                predictions.loc[predictions["valid_date"] == unique_date], 
                                                                                                x_name,
                                                                                                y_name)
        return predictions

