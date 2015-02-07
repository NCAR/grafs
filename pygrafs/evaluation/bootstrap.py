import numpy as np
from scipy.stats import scoreatpercentile as sap

def bootstrap(forecast, observed, score, n_boot=1000, **kwargs):
    indices = np.random.randint(0, forecast.size, (n_boot, forecast.size))
    boot_fore = forecast[indices]
    boot_obs = observed[indices]
    scores = np.array([score(boot_fore[i], boot_obs[i], **kwargs) for i in range(n_boot)])
    return scores

def bootstrap_ci(scores, lower_percentile, upper_percentile):
    return np.array([sap(scores, lower_percentile), sap(scores, upper_percentile)])