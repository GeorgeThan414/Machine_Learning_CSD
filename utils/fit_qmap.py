import os
import numpy as np
import pandas as pd
import torch

from model.mlp import MLPRegressor
from utils.build_loaders import build_loaders
from utils.testing_loop import predict_households
from utils.quantile_mapping import fit_quantile_mapping

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- paths
    run_dir = "runs/mlp_ensemble"
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_csv = "data/train.csv"
    train_gt_csv = "data/train_hh_gt.csv"

    # We'll fit mapping using ONLY survey 300000 (thresholds derived from it)
    valid_survey_ids = [300000]

    # load GT consumption to get true cons for survey 300000
    gt = pd.read_csv(train_gt_csv)
    gt = gt[gt["survey_id"].isin(valid_survey_ids)].copy()
    true_map = {(int(r.survey_id), int(r.hhid)): float(r.cons_ppp17) for r in gt.itertuples(index=False)}

    # build loader that returns those households
    _, val_loader, _ = build_loaders(
        train_csv=train_csv,
        train_gt_csv=train_gt_csv,
        test_csv=train_csv,   # dummy valid
        batch_size=512,
        seed=42,
        valid_survey_ids=valid_survey_ids,
        max_valid_samples=None,
    )

    # --- load ensemble checkpoints (you will create them in training)
    ckpts = [
        os.path.join(ckpt_dir, "best_seed0.pth"),
        os.path.join(ckpt_dir, "best_seed1.pth"),
        os.path.join(ckpt_dir, "best_seed2.pth"),
    ]
    ckpts = [p for p in ckpts if os.path.exists(p)]
    if len(ckpts) == 0:
        raise FileNotFoundError("No ensemble checkpoints found. Train first.")

    # model config MUST match checkpoints
    model = MLPRegressor(
        input_dim=26,
        hidden_dims=(128, 256, 128, 64),
        dropout=0.1,
        use_residual=True,
    ).to(device)
    model.eval()

    # --- predict log-space for each model, average log-space
    all_log = []
    sids_ref = None
    hhids_ref = None

    for p in ckpts:
        state = torch.load(p, map_location=device)
        model.load_state_dict(state)
        preds, sids, hhids = predict_households(model, val_loader, device) 

        
       
        if np.nanmedian(preds) > 50:  # heuristic: real space can be large
            logp = np.log1p(np.clip(preds, 0.0, None))
        else:
            # maybe it's already log-space
            logp = preds

        all_log.append(logp.astype(np.float64))
        if sids_ref is None:
            sids_ref = sids
            hhids_ref = hhids

    log_mean = np.mean(np.stack(all_log, axis=0), axis=0)
    pred_real = np.expm1(log_mean).astype(np.float32)

    # --
    true_real = np.array([true_map[(int(sids_ref[i]), int(hhids_ref[i]))] for i in range(len(hhids_ref))], dtype=np.float32)

    mapping = fit_quantile_mapping(pred_real, true_real, n_quantiles=2000)

    out_path = os.path.join(ckpt_dir, "qmap_300000.npz")
    np.savez(out_path, x=mapping["x"], y=mapping["y"])
    print("Saved quantile mapping:", out_path)

