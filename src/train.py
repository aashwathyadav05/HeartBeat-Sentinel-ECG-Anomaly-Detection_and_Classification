import os
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .preprocess import prepare_dataloaders, set_seed
from .model import ConvAutoencoder, count_params


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def train(epochs: int = 50, lr: float = 1e-3, batch_size: int = 64, patience: int = 10):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    info = prepare_dataloaders(batch_size=batch_size)
    seq_len = info["seq_len"]
    train_loader = info["train_loader"]
    val_loader = info["val_loader"]

    model = ConvAutoencoder(seq_len=seq_len, latent_dim=32).to(device)
    print(f"Model params: {count_params(model)}")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val = float('inf')
    best_path = os.path.join(MODEL_DIR, "cae_best.pt")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            xb = xb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, xb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                xb = xb.to(device)
                out = model(xb)
                loss = criterion(out, xb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "seq_len": seq_len,
                "latent_dim": 32,
                "history": history,
            }, best_path)
            print(f"Saved best model to {best_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    with open(os.path.join(MODEL_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Reconstruction Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "loss_curves.png"))
    plt.close()
    print("Done.")


if __name__ == "__main__":
    train()
