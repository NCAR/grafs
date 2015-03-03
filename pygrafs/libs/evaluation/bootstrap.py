import numpy as np
from scipy.stats import scoreatpercentile as sap

def bootstrap(forecast, observed, score, n_boot=1000, **kwargs):
    """
    Calculate a bootstrap sampling of the values of a particular score

    :param forecast: Array of forecast values
    :param observed: Array of observations
    :param score: Score function
    :param n_boot: Number of bootstrap iterations
    :param kwargs: Additional keyword arguments required for a particular score function
    :return:
    """
    indices = np.random.randint(0, forecast.size, (n_boot, forecast.size))
    boot_fore = forecast[indices]
    boot_obs = observed[indices]
    scores = np.array([score(boot_fore[i], boot_obs[i], **kwargs) for i in range(n_boot)])
    return scores


def bootstrap_ci(scores, lower_percentile, upper_percentile):
    """
    Retrieve a confidence interval for a set of bootstrap scores

    :param scores: Set of scores retrieved from the bootstrap function.
    :param lower_percentile: Lower bound percentile (5=5th percentile)
    :param upper_percentile: Upper bound percentile (95 = 95th percentile)
    :return: Array containing the lower and upper percentiles
    """
    return np.array([sap(scores, lower_percentile), sap(scores, upper_percentile)])