from tqdm import tqdm
import torch


def validate_one_epoch(model,loader,criterion,device,):
    """
    Validation loop for tabular regression.
    Returns:
        avg_val_loss (float)
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="Validation", leave=False)

    with torch.no_grad():
        for x, y, _, _ in pbar:
            x = x.to(device)
            y = y.to(device)

            pred = model(x).squeeze()   # [B]
            loss = criterion(pred, y)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            pbar.set_postfix({"val_loss": f"{loss.item():.6f}"})

    avg_val_loss = total_loss / total_samples
    return avg_val_loss
