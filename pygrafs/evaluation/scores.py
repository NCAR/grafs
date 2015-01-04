import numpy as np


def mean_error(forecast, observed):
    return np.mean(forecast - observed)


def mean_absolute_error(forecast, observed):
    return np.mean(np.abs(forecast - observed))


def mean_squared_error(forecast, observed):
    return np.mean((forecast - observed) ** 2)


def root_mean_squared_error(forecast, observed):
    return np.sqrt(mean_squared_error(forecast, observed))
