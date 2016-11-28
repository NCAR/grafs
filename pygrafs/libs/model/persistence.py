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

    def make_predictions(self, valid_dates, train_sites, test_sites, x_name="lon", y_name="lat"):
        valid_dates_index = pd.DatetimeIndex(valid_dates)
        start_dates = valid_dates_index - pd.Timedelta(self.offset_hours, unit="h")
        obs = ObsSeries(self.variable, start_dates, self.lon_bounds, self.lat_bounds, self.obs_path)
        obs.load_data()
        unique_dates = valid_dates_index.unique()
        predictions = pd.DataFrame(obs.data.loc[np.in1d(obs.data["station"], test_sites),
                                                ["valid_date", "station", x_name, y_name]])
        predictions["persistence"] = 0
        for date in unique_dates:
            predictions.loc[predictions["valid_date"] == date,
                            "persistence"] = cressman(obs.data.loc[np.in1d(obs.data["station"], train_sites)
                                                                   & (obs.data["valid_date"] == date)],
                                                      obs.data.loc[np.in1d(obs.data["station"], test_sites)
                                                                   & (obs.data["valid_date"] == date)],
                                                      x_name,
                                                      y_name)
        return predictions

