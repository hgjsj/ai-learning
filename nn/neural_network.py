"""Neural network demo using PyTorch.

This file contains a self‑contained example showing how to:
1. Create a synthetic regression dataset.
2. Define a simple Multi-Layer Perceptron (MLP).
3. Train for a few epochs and report losses.
4. Show sample predictions.

Run from project root:
    python main.py --demo --epochs 3

Or directly:
    python -m nn.demo --epochs 3

All code is intentionally small & CPU friendly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "DemoConfig",
    "make_synthetic_regression",
    "build_model",
    "train_one_epoch",
    "evaluate",
    "train_demo",
    "run_demo",
]


# ---------------------------------------------------------------------------
# Configuration & utilities
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class DemoConfig:
    input_dim: int = 5
    hidden_dim: int = 32
    output_dim: int = 1
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-2
    train_size: int = 512
    val_size: int = 128
    seed: int = 42
    device: str | None = None  # Autodetect if None

    def resolve_device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def make_synthetic_regression(n: int, input_dim: int, noise_std: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a simple non-linear regression dataset.

    y = (W x + b) + 0.5 * sin(x0) + noise
    """
    x = torch.randn(n, input_dim)
    W = torch.linspace(1.0, -1.0, input_dim)
    b = 0.5
    linear = (x * W).sum(dim=1, keepdim=True) + b
    nonlinear = 0.5 * torch.sin(x[:, 0:1])
    y = linear + nonlinear + noise_std * torch.randn_like(linear)
    return x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(cfg: DemoConfig) -> nn.Module:
    return nn.Sequential(
        nn.Linear(cfg.input_dim, cfg.hidden_dim),
        nn.ReLU(),
        nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        nn.ReLU(),
        nn.Linear(cfg.hidden_dim, cfg.output_dim),
    )


# ---------------------------------------------------------------------------
# Training & evaluation loops
# ---------------------------------------------------------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, loss_fn, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def train_demo(cfg: DemoConfig | None = None) -> dict:
    start_time = time.time()
    cfg = cfg or DemoConfig()
    set_seed(cfg.seed)
    device = cfg.resolve_device()

    x_train, y_train = make_synthetic_regression(cfg.train_size, cfg.input_dim)
    x_val, y_val = make_synthetic_regression(cfg.val_size, cfg.input_dim)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=cfg.batch_size)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    history: list[tuple[float, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        history.append((train_loss, val_loss))
        print(f"Epoch {epoch:02d}/{cfg.epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed:.2f}s. Final val_loss={history[-1][1]:.4f}")

    return {
        "cfg": cfg,
        "final_train_loss": history[-1][0],
        "final_val_loss": history[-1][1],
        "model": model,
        "history": history,
    }


def run_demo():  # pragma: no cover - thin wrapper
    results = train_demo()
    model = results["model"]
    cfg: DemoConfig = results["cfg"]
    device = cfg.resolve_device()
    model.eval()
    with torch.no_grad():
        x_test, y_test = make_synthetic_regression(5, cfg.input_dim)
        preds = model(x_test.to(device)).cpu().squeeze().tolist()
        targets = y_test.squeeze().tolist()
        print("Sample predictions (pred -> target):")
        for p, t in zip(preds, targets):
            print(f"  {p:.3f} -> {t:.3f}")
    return results


# ---------------------------------------------------------------------------
# Allow module execution: python -m nn.demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    run_demo()

