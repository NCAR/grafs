from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator, Rbf
from sklearn.gaussian_process import GaussianProcess
from scipy.spatial.distance import cdist
import numpy as np
__author__ = 'djgagne2'


def nearest_neighbor(predictions, grid_coordinates, y_name, x_name):
    """
       Nearest Neighbor interpolation.

       :param predictions: pandas DataFrame containing the x and y coordinates of each training site and the prediction
           value in the last column.
       :param grid_coordinates: pandas dataframe containing the x and y coordinates of each grid point
       :param y_name: Name of the y-coordinate column
       :param x_name: Name of the x-coordinate column
       :return: A flat array of predictions at each grid point.
       """
    nn = NearestNDInterpolator(predictions[[y_name, x_name]].values, predictions.ix[:, -1].values)
    grid_predictions = nn(np.vstack((grid_coordinates[y_name].values, grid_coordinates[x_name].values)).T)
    return grid_predictions


def linear_interpolation(predictions, grid_coordinates, y_name, x_name):
    lin = LinearNDInterpolator(predictions[[y_name, x_name]].values, predictions.ix[:, -1].values)
    grid_predictions = lin(np.vstack((grid_coordinates[y_name].ravel(), grid_coordinates[x_name].ravel())).T)
    return grid_predictions


def rbf_interpolation(predictions, grid_coordinates, y_name, x_name, method="gaussian"):
    rbf = Rbf(predictions[y_name].values, predictions[x_name].values, predictions.ix[:, -1], method=method)
    grid_predictions = rbf(grid_coordinates[y_name].ravel(), grid_coordinates[x_name].ravel())
    return grid_predictions


def gaussian_process_interpolation(predictions, grid_coordinates, y_name, x_name):
    gp = GaussianProcess(regr='linear', nugget=0.01)
    gp.fit(predictions[[y_name, x_name]].values, predictions)
    grid_predictions = gp.predict(np.vstack([grid_coordinates[y_name].ravel(),
                                             grid_coordinates[x_name].ravel()]).T)
    return grid_predictions


def cressman(predictions, grid_coordinates, y_name, x_name, radii=None):
    """
    Cressman successive correction interpolation.

    :param predictions: pandas DataFrame containing the x and y coordinates of each training site and the prediction
        value in the last column.
    :param grid_coordinates: pandas dataframe containing the x and y coordinates of each grid point
    :param y_name: Name of the y-coordinate column
    :param x_name: Name of the x-coordinate column
    :param radii: sequence of maximum radii for each iteration of the interpolation. The radii should be applied in
        decreasing order. If None, radii will be determined based on the distribution of distances.

    :return: A flat array of predictions at each grid point.
    """
    distances = cdist(grid_coordinates[[x_name, y_name]], predictions[[x_name, y_name]], metric='sqeuclidean')
    if radii is None:
        radii = np.percentile(np.sqrt(distances.ravel()), np.array([90, 75, 50, 25]))
    train_predictions = predictions.ix[:, -1].values.reshape((1, predictions.shape[0]))
    grid_predictions = np.ones((grid_coordinates.shape[0], 1)) * train_predictions.mean()
    for radius in radii:
        weights = (radius ** 2 - distances) / (radius ** 2 + distances)
        weights[weights < 0] = 0
        grid_predictions[:, 0] += np.sum(weights * (train_predictions - grid_predictions), axis=1) / weights.sum(axis=1)
    return grid_predictions
