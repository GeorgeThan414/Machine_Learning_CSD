import numpy as np
import pandas as pd

from utils.metrics import compute_poverty_rates, poverty_thresholds, threshold_weights
from utils.testing_loop import predict_households
from utils.build_loaders import build_loaders
from model.mlp import MLPRegressor
import torch
import os


# Load model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MLPRegressor(
    input_dim=26,
    hidden_dims=(128,256,128,64),
    dropout=0.1,
    use_residual=True,
)
model.load_state_dict(torch.load("runs/mlp_baseline/checkpoints/best.pth", map_location=device))
model.to(device)
model.eval()


# Load data
train_csv = "data/train.csv"
train_gt_csv = "data/train_hh_gt.csv"
raw_train_features_csv = "data/train_hh_features.csv"
rates_gt_csv = "data/train_rates_gt.csv"

train_loader, val_loader, _ = build_loaders(
    train_csv=train_csv,
    train_gt_csv=train_gt_csv,
    test_csv=train_csv,
    batch_size=256,
    seed=42,
    valid_survey_ids=[300000],
    max_valid_samples=None,
)


# Predict on TRAIN surveys
preds, sids, hhids = predict_households(model, val_loader, device)


# Load weights + true rates
# Load weights
weights_df = pd.read_csv(
    raw_train_features_csv,
    usecols=["survey_id", "hhid", "weight"]
)

weights_map = {
    (int(r.survey_id), int(r.hhid)): float(r.weight)
    for r in weights_df.itertuples(index=False)
}

# Load true poverty rates
true_rates_df = pd.read_csv(rates_gt_csv)

rate_cols = [c for c in true_rates_df.columns if c.startswith("pct_hh_below_")]

true_rates_map = {}
for _, row in true_rates_df.iterrows():
    sid = int(row["survey_id"])
    true_rates_map[sid] = row[rate_cols].values.astype(np.float32)


for sid in sorted(true_rates_map.keys()):
    idx = np.where(sids == sid)[0]
    cons = preds[idx]
    w = np.array([weights_map[(sid,int(hhids[i]))] for i in idx])

    pred_rates = compute_poverty_rates(cons, w)
    true_rates = true_rates_map[sid]

    df = pd.DataFrame({
        "threshold": poverty_thresholds,
        "true_rate": true_rates,
        "pred_rate": pred_rates,
        "abs_error": np.abs(pred_rates - true_rates),
        "rel_error": np.abs(pred_rates - true_rates) / (true_rates + 1e-8),
        "weight": threshold_weights,
    })

    print(f"\n===== Survey {sid} =====")
    print(df.to_string(index=False))
    print("Weighted poverty wMAPE:",
          np.mean(df["rel_error"] * df["weight"]))
