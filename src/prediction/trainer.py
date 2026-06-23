# src/prediction/trainer.py
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger

from src.prediction.transformer_model import build_model
from src.utils.eth_ucyLoader import ETHUCYDataset, load_all_subsets
from src.utils.logger import setup_logger


def ade_loss(pred, target):
    return torch.sqrt(((pred - target) ** 2).sum(dim=-1)).mean()

def fde_loss(pred, target):
    return torch.sqrt(((pred[:, -1, :] - target[:, -1, :]) ** 2).sum(dim=-1)).mean()


class Trainer:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.cfg    = self.config["model"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on: {self.device}")

        self.model = build_model(self.config).to(self.device)
        logger.info(f"Parameters: {self.model.count_parameters():,}")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg["learning_rate"], weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=10, factor=0.5
        )
        self.train_losses = []
        self.val_losses   = []

    def load_data(self, data_path: str):
        traj_cfg = self.config["trajectory"]
        path = Path(data_path)

        if path.is_file():
            dataset = ETHUCYDataset(
                filepath  = str(path),
                obs_len   = traj_cfg["obs_len"],
                pred_len  = traj_cfg["pred_len"],
                skip      = traj_cfg["skip"],
                normalize = True,
            )
        else:
            import numpy as np
            obs, pred = load_all_subsets(
                str(path),
                obs_len  = traj_cfg["obs_len"],
                pred_len = traj_cfg["pred_len"],
                skip     = traj_cfg["skip"],
            )
            dataset = TensorDataset(
                torch.tensor(obs,  dtype=torch.float32),
                torch.tensor(pred, dtype=torch.float32),
            )

        n_val   = max(1, int(0.2 * len(dataset)))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])

        self.train_loader = DataLoader(train_ds, batch_size=self.cfg["batch_size"], shuffle=True,  num_workers=0)
        self.val_loader   = DataLoader(val_ds,   batch_size=self.cfg["batch_size"], shuffle=False, num_workers=0)
        logger.info(f"Train: {n_train} | Val: {n_val} sequences")

    def train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        for obs, target in self.train_loader:
            obs, target = obs.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(obs, teacher_forcing=target)
            loss = ade_loss(pred, target)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total += loss.item()
        return total / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_ade = total_fde = 0.0
        for obs, target in self.val_loader:
            obs, target = obs.to(self.device), target.to(self.device)
            pred = self.model.predict(obs)
            total_ade += ade_loss(pred, target).item()
            total_fde += fde_loss(pred, target).item()
        n = len(self.val_loader)
        return total_ade / n, total_fde / n

    def save_checkpoint(self, epoch, val_ade, is_best=False):
        save_dir = Path(self.config["paths"]["models"]) / "predictor"
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "epoch": epoch, "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_ade": val_ade, "config": self.config,
        }
        torch.save(ckpt, save_dir / "checkpoint_latest.pt")
        if is_best:
            torch.save(ckpt, save_dir / "checkpoint_best.pt")
            logger.info(f"  ✅ Best model saved (ADE={val_ade:.4f})")

    def plot_losses(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.train_losses, label="Train ADE", color="steelblue")
        ax.plot(self.val_losses,   label="Val ADE",   color="coral")
        ax.set_xlabel("Epoch"); ax.set_ylabel("ADE Loss")
        ax.set_title("Trajectory Transformer — Training Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
        out = Path("outputs/visualizations/training_loss.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Loss plot → {out}")

    def train(self, data_path: str):
        self.load_data(data_path)
        epochs     = self.cfg["epochs"]
        ckpt_every = self.cfg["checkpoint_every"]
        best_val   = float("inf")

        logger.info(f"\n{'='*55}")
        logger.info(f"  Starting training | {epochs} epochs")
        logger.info(f"{'='*55}")

        for epoch in range(1, epochs + 1):
            train_loss       = self.train_epoch()
            val_ade, val_fde = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_ade)
            self.scheduler.step(val_ade)

            is_best = val_ade < best_val
            if is_best:
                best_val = val_ade

            logger.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train ADE: {train_loss:.4f} | "
                f"Val ADE: {val_ade:.4f} | "
                f"Val FDE: {val_fde:.4f}"
                + (" ← best" if is_best else "")
            )

            if epoch % ckpt_every == 0 or is_best:
                self.save_checkpoint(epoch, val_ade, is_best)

        self.plot_losses()
        logger.info(f"\nTraining complete! Best Val ADE: {best_val:.4f}")
        logger.info(f"Model → models/predictor/checkpoint_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    setup_logger()
    trainer = Trainer(config_path=args.config)
    trainer.train(data_path=args.data)