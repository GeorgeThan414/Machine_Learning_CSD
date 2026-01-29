import numpy as np

# Poverty thresholds (fixed by competition)
poverty_thresholds = np.array([
    3.17, 3.94, 4.60, 5.26, 5.88,
    6.47, 7.06, 7.70, 8.40, 9.13,
    9.87, 10.70, 11.62, 12.69, 14.03,
    15.64, 17.76, 20.99, 27.37
], dtype=np.float32)

# Corresponding ventiles (given)
ventiles = np.array([
     5, 10, 15, 20, 25,
    30, 35, 40, 45, 50,
    55, 60, 65, 70, 75,
    80, 85, 90, 95
], dtype=np.float32)

# Weights emphasize thresholds near 40%
threshold_weights = np.exp(-np.abs(ventiles - 40) / 10.0)
threshold_weights /= threshold_weights.sum()


# Household-level metric
def household_mape(pred, true, eps=1e-8):
    """
    pred, true: (N,)
    """
    return np.mean(np.abs(pred - true) / (true + eps))



# Poverty rate computation for ONE survey
def compute_poverty_rates(consumption, weights):
    """
    consumption: (N,) cons_ppp17
    weights:     (N,) population weights

    returns: (19,) poverty rates
    """
    rates = []
    total_w = weights.sum()

    for thr in poverty_thresholds:
        mask = consumption < thr
        rate = weights[mask].sum() / total_w
        rates.append(rate)

    return np.array(rates, dtype=np.float32)



# Weighted poverty-rate MAPE (main competition metric)
def weighted_poverty_mape(pred_rates, true_rates, eps=1e-8):
    """
    pred_rates, true_rates: (num_surveys, 19)
    """
    mape = np.abs(pred_rates - true_rates) / (true_rates + eps)
    weighted = mape * threshold_weights[None, :]
    return weighted.mean()
