import numpy as np


def mean_error(forecast, observed):
    """
    The mean difference between the forecast and observed values.

    :param forecast: Array of forecast values
    :param observed: Array of observations
    :return: The score
    """
    return np.mean(forecast - observed)


def mean_absolute_error(forecast, observed):
    """
    The mean of the absolute values of the differences between the forecasts and observations.

    :param forecast: Array of forecast values
    :param observed: Array of observations
    :return: The score
    """
    return np.mean(np.abs(forecast - observed))


def mean_squared_error(forecast, observed):
    """
    The mean of the squared differences between the forecasts and observations.

    :param forecast: Array of forecast values
    :param observed: Array of observations
    :return:
    """
    return np.mean((forecast - observed) ** 2)


def root_mean_squared_error(forecast, observed):
    """
    The square root of the mean squared error.

    :param forecast:
    :param observed:
    :return:
    """
    return np.sqrt(mean_squared_error(forecast, observed))
