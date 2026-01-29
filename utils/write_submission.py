import os
import zipfile
import numpy as np
import pandas as pd
import torch

from utils.build_loaders import build_loaders
from utils.testing_loop import predict_households
from utils.metrics import compute_poverty_rates, poverty_thresholds
from model.mlp import MLPRegressor
from torch.utils.data import DataLoader
from utils.datasets import HouseholdDataset

def write_submission():
    # Paths
    data_dir = "data"
    run_dir = "runs/mlp_baseline"
    out_dir = "submission"

    os.makedirs(out_dir, exist_ok=True)

    test_csv = os.path.join(data_dir, "test.csv")
    test_features_csv = os.path.join(data_dir, "test_hh_features.csv")
    checkpoint_path = os.path.join(run_dir, "checkpoints", "best.pth")

   
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    # Load model
    model = MLPRegressor(
        input_dim=26,
        hidden_dims=(128, 256, 128, 64),
        dropout=0.1,
        use_residual=True,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    print("Model loaded.")

    # Build test loader (inference only)
    test_dataset = HouseholdDataset(
    features_csv=test_csv,
    gt_csv=None,           
)

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,)

    # Predict household consumption
    preds, survey_ids, hhids = predict_households(model, test_loader, device)

    household_df = pd.DataFrame({
        "survey_id": survey_ids.astype(int),
        "hhid": hhids.astype(int),
        "cons_ppp17": preds.astype(float),
    })

    household_csv = os.path.join(out_dir, "predicted_household_consumption.csv")
    household_df.to_csv(household_csv, index=False)
    print(f"Saved: {household_csv}")

 
    # Load weights (population weights)
    weights_df = pd.read_csv(
    test_features_csv,
    usecols=["survey_id", "hhid", "weight"],
    )

    weights_map = {
        (int(r.survey_id), int(r.hhid)): float(r.weight)
        for r in weights_df.itertuples(index=False)
    }

    # Compute poverty distribution
    poverty_rows = []

    for sid in sorted(household_df["survey_id"].unique()):
        sub = household_df[household_df["survey_id"] == sid]

        cons = sub["cons_ppp17"].values
        weights = np.array(
            [weights_map[(sid, int(h))] for h in sub["hhid"]],
            dtype=np.float32,
        )

        rates = compute_poverty_rates(cons, weights)

        row = {"survey_id": int(sid)}
        for i, thr in enumerate(poverty_thresholds):
            row[f"pct_hh_below_{thr:.2f}"] = float(rates[i])

        poverty_rows.append(row)

    poverty_df = pd.DataFrame(poverty_rows)

    poverty_csv = os.path.join(out_dir, "predicted_poverty_distribution.csv")
    poverty_df.to_csv(poverty_csv, index=False)
    print(f"Saved: {poverty_csv}")


    # Zip submission
    zip_path = os.path.join(out_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(household_csv, arcname="predicted_household_consumption.csv")
        z.write(poverty_csv, arcname="predicted_poverty_distribution.csv")

    print(f"\n Submission ready: {zip_path}")


if __name__ == "__main__":
    write_submission()
