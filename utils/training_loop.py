from tqdm import tqdm
import torch


def train_one_epoch(model,loader,optimizer,criterion,device):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Train", leave=False)

    for x, y, _, _ in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x).squeeze()      # [B]
        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    return total_loss / len(loader.dataset)
