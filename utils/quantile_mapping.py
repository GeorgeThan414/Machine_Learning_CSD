import numpy as np

def fit_quantile_mapping(pred, true, n_quantiles=2000, eps=1e-8):
    """
    Fit monotonic mapping: pred -> true using quantiles.
    Returns a dict with arrays 'x' (pred_q) and 'y' (true_q).

    pred,true: (N,) in REAL consumption space (>=0 ideally).
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)

    pred = np.clip(pred, 0.0, None)
    true = np.clip(true, 0.0, None)

    # remove NaNs / infs
    m = np.isfinite(pred) & np.isfinite(true)
    pred = pred[m]
    true = true[m]

    q = np.linspace(0.0, 1.0, n_quantiles)

    pred_q = np.quantile(pred, q)
    true_q = np.quantile(true, q)

    # ensure strictly increasing x for interpolation (collapse duplicates)
    # If many duplicates in pred_q, np.interp gets unstable.
    x = pred_q
    y = true_q

    # make x non-decreasing strictly by tiny jitter
    dx = np.diff(x)
    if np.any(dx <= 0):
        # enforce monotonic increasing with eps spacing
        x_fixed = x.copy()
        for i in range(1, len(x_fixed)):
            if x_fixed[i] <= x_fixed[i-1]:
                x_fixed[i] = x_fixed[i-1] + eps
        x = x_fixed

    return {"x": x.astype(np.float64), "y": y.astype(np.float64)}


def apply_quantile_mapping(pred, mapping):
    """
    Apply fitted quantile mapping to predictions (REAL consumption space).
    pred: (N,)
    mapping: dict from fit_quantile_mapping
    """
    pred = np.asarray(pred, dtype=np.float64)
    pred = np.clip(pred, 0.0, None)

    x = mapping["x"]
    y = mapping["y"]

    # piecewise-linear monotonic map with extrapolation by endpoints
    return np.interp(pred, x, y, left=y[0], right=y[-1]).astype(np.float32)
