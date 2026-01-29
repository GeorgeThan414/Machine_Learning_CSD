import torch
import numpy as np
from tqdm import tqdm


"""
# Predict household consumption
def predict_households(model, loader, device):
    
    Returns:
        preds      : (N,)
        survey_ids : (N,)
        hhids      : (N,)

    model.eval()

    preds = []
    survey_ids = []
    hhids = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict", leave=False):
            if len(batch) == 4:
                x, _, sid, hid = batch
            else:
                x, sid, hid = batch

            x = x.to(device)
            out = model(x).squeeze()

            # inverse log -> real consumption
            out = torch.expm1(out)

            preds.append(out.cpu().numpy())
            survey_ids.extend(sid)
            hhids.extend(hid)

    preds = np.concatenate(preds, axis=0)

    return (
        preds,
        np.array(survey_ids, dtype=np.int64),
        np.array(hhids, dtype=np.int64),
    )
"""

def predict_households(model, loader, device, return_log=True):
    """
    If return_log=True: returns log1p(cons) predictions (N,)
    If return_log=False: returns real cons predictions (N,)
    """
    model.eval()

    preds = []
    survey_ids = []
    hhids = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict", leave=False):
            if len(batch) == 4:
                x, _, sid, hid = batch
            else:
                x, sid, hid = batch

            x = x.to(device)
            out = model(x).squeeze(1) if model(x).dim() == 2 and model(x).shape[1] == 1 else model(x).squeeze()
            out = out.detach().cpu().numpy().astype(np.float32)

            if return_log:
                preds.append(out)
            else:
                preds.append(np.expm1(out))

            survey_ids.extend(sid.tolist() if hasattr(sid, "tolist") else list(sid))
            hhids.extend(hid.tolist() if hasattr(hid, "tolist") else list(hid))

    preds = np.concatenate(preds, axis=0)
    return preds, np.array(survey_ids, dtype=np.int64), np.array(hhids, dtype=np.int64)