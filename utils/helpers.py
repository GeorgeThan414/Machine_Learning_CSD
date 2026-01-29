import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.metrics import compute_poverty_rates, weighted_poverty_mape
from utils.testing_loop import predict_households


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_loss_plot(train_losses, val_losses, out_path, title):
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def load_weights_map_from_raw(raw_csv_path: str):
    """
    Returns dict:
        weights_map[(survey_id, hhid)] = weight
    """
    df = pd.read_csv(raw_csv_path, usecols=["survey_id", "hhid", "weight"])
    # ensure numeric
    df["survey_id"] = df["survey_id"].astype(int)
    df["hhid"] = df["hhid"].astype(int)
    df["weight"] = df["weight"].astype(float)
    return {(int(r.survey_id), int(r.hhid)): float(r.weight) for r in df.itertuples(index=False)}


def load_true_rates_map(train_rates_gt_csv: str):
    """
    Returns dict:
        true_rates_map[survey_id] = np.array([19 rates], dtype=float)
    """
    df = pd.read_csv(train_rates_gt_csv)
    df["survey_id"] = df["survey_id"].astype(int)
    rate_cols = [c for c in df.columns if c.startswith("pct_hh_below_")]
    true_rates_map = {}
    for r in df.itertuples(index=False):
        sid = int(getattr(r, "survey_id"))
        true_rates_map[sid] = np.array([getattr(r, c) for c in rate_cols], dtype=np.float32)
    return true_rates_map, rate_cols


def poverty_metric_on_loader(
    model,
    loader,
    device,
    weights_map,
    true_rates_map,
):
    """
    Computes competition poverty weighted MAPE on the surveys present in loader,
    provided that those survey_ids exist in true_rates_map (training surveys only).
    """
    preds, survey_ids, hhids = predict_households(model, loader, device)

    # group predictions by survey, collect weights aligned to prediction order
    pred_rates_list = []
    true_rates_list = []

    unique_surveys = sorted(set(survey_ids.tolist()))
    for sid in unique_surveys:
        if sid not in true_rates_map:
            continue  # e.g., if loader has surveys without GT rates

        idx = np.where(survey_ids == sid)[0]
        cons = preds[idx].astype(np.float32)

        w = np.array([weights_map[(int(sid), int(hhids[i]))] for i in idx], dtype=np.float32)

        pr = compute_poverty_rates(consumption=cons, weights=w)
        tr = true_rates_map[sid]

        pred_rates_list.append(pr)
        true_rates_list.append(tr)

    if len(pred_rates_list) == 0:
        return np.nan

    pred_rates = np.stack(pred_rates_list, axis=0)
    true_rates = np.stack(true_rates_list, axis=0)
    return float(weighted_poverty_mape(pred_rates, true_rates))