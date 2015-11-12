from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator, Rbf
import numpy as np
__author__ = 'djgagne2'


def nearest_neighbor(predictions, grid_coordinates, y_name, x_name):
    nn = NearestNDInterpolator(predictions[[y_name, x_name]].values, predictions.ix[:, -1].values)
    grid_predictions = nn(np.vstack((grid_coordinates[y_name].values, grid_coordinates[x_name].values)).T)
    return grid_predictions


def linear_interpolation(predictions, grid_coordinates, y_name, x_name):
    print predictions.shape, grid_coordinates.shape, y_name, x_name
    lin = LinearNDInterpolator(predictions[[y_name, x_name]].values, predictions.ix[:, -1].values)
    grid_predictions = lin(np.vstack((grid_coordinates[y_name].ravel(), grid_coordinates[x_name].ravel())).T)
    return grid_predictions


def rbf_interpolation(predictions, grid_coordinates, y_name, x_name, method="gaussian"):
    rbf = Rbf(predictions[y_name].values, predictions[x_name].values, predictions.ix[:, -1], method=method)
    grid_predictions = rbf(grid_coordinates[y_name].ravel(), grid_coordinates[x_name].ravel())
    return grid_predictions
