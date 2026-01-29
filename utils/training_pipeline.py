from model.mlp import MLPRegressor
import os
import time
import copy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from utils.build_loaders import build_loaders
from utils.metrics import compute_poverty_rates, weighted_poverty_mape
from utils.training_loop import train_one_epoch
from utils.validation_loop import validate_one_epoch 
from utils.testing_loop import predict_households
from utils.helpers import ensure_dir
from utils.helpers import save_loss_plot
from utils.helpers import load_true_rates_map
from utils.helpers import load_weights_map_from_raw
from utils.helpers import poverty_metric_on_loader
from utils.seed import set_seed

def run (
    model, seed):
    
    data_dir = "data"
    run_dir = "runs/mlp_baseline"

    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")
    train_gt_csv = os.path.join(data_dir, "train_hh_gt.csv")
    train_rates_gt_csv = os.path.join(data_dir, "train_rates_gt.csv")
    raw_train_features_csv = os.path.join(data_dir, "train_hh_features.csv")

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    plot_dir = os.path.join(run_dir, "plots")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    # Data loaders Parameters
    batch_size = 128
    valid_survey_ids = [300000]
    max_valid_samples = 10_000

    train_loader, val_loader, test_loader = build_loaders(
        train_csv=train_csv,
        train_gt_csv=train_gt_csv,
        test_csv=test_csv,
        batch_size=batch_size,
        seed=seed,
        valid_survey_ids=valid_survey_ids,
        max_valid_samples=max_valid_samples,
    )

    # Hyperparameters
    epochs = 30
    learning_rate = 1e-3
    weight_decay = 1e-4
    scheduler_gamma = 0.5
    early_patience = 5
    min_delta = 1e-5

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=scheduler_gamma)    
    criterion=  nn.L1Loss()


    weights_df = pd.read_csv(
        raw_train_features_csv,
        usecols=["survey_id", "hhid", "weight"],)

    weights_map = {
        (int(r.survey_id), int(r.hhid)): float(r.weight)
        for r in weights_df.itertuples(index=False)}

    rates_df = pd.read_csv(train_rates_gt_csv)
    rate_cols = [c for c in rates_df.columns if c.startswith("pct_hh_below_")]
    true_rates_map = {
    int(row["survey_id"]): row[rate_cols].values.astype(np.float32)
    for _, row in rates_df.iterrows()
}

    best_val_loss = float("inf")
    best_state = None
    patience = 0

    train_losses = []
    val_losses = []
    val_poverty_scores = []

    start_time = time.time()
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        #compute poverty metric on val_loader
        preds, sids, hhids = predict_households(model, val_loader, device)

        pred_rates = []
        true_rates = []

        for sid in set(sids.tolist()):
            if sid not in true_rates_map:
                continue
            idx = np.where(sids == sid)[0]
            cons = preds[idx]
            w = np.array(
                [weights_map[(sid, int(hhids[i]))] for i in idx],
                dtype=np.float32,
            )
            pred_rates.append(compute_poverty_rates(cons, w))
            true_rates.append(true_rates_map[sid])

        poverty_mape = (
            weighted_poverty_mape(
                np.stack(pred_rates),
                np.stack(true_rates),
            )
            if pred_rates else np.nan
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_poverty_scores.append(poverty_mape)

        scheduler.step()

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_poverty_wMAPE={poverty_mape:.6f}"
        )

        # early stopping (by val loss) 
        if best_val_loss - val_loss > min_delta:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                        best_state,
                        os.path.join(ckpt_dir, f"best_seed{seed}.pth"))
            patience = 0
        else:
            patience += 1

        if patience >= early_patience:
            print("Early stopping triggered.")
            break
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f}s")
   
    torch.save(model.state_dict(),os.path.join(ckpt_dir, f"last_seed{seed}.pth"))
    if best_state is not None:
        model.load_state_dict(best_state)
    
    save_loss_plot(
        train_losses,
        val_losses,
        os.path.join(plot_dir, "loss_plot.png"),
        title="train vs val loss")

    summary = {
    "epochs_ran": int(len(train_losses)),
    "best_val_loss": float(best_val_loss),
    "final_val_poverty_wMAPE": float(val_poverty_scores[-1]),
    "device": str(device),
    "batch_size": int(batch_size),
    "elapsed_time_sec": float(time.time() - start_time),
}

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Training completed in {time.time() - start_time:.2f}s")
    print("Summary:", summary)

    return model, summary



if __name__ == "__main__":
    for seed in [0,1,2]:
        print(f"\n===== TRAINING SEED {seed} =====\n")
        model = MLPRegressor(
            input_dim=26,
            hidden_dims=(128, 256, 128, 64),
            dropout=0.1,
            use_residual=True,
        )

        run(model, seed=seed)
