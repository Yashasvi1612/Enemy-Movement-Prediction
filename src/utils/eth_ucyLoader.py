
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset
import torch
from loguru import logger


# ── Raw data loader ───────────────────────────────────────────────────────────
def load_eth_ucy_file(filepath: str) -> pd.DataFrame:
    """
    Load a single ETH/UCY annotation file into a DataFrame.

    Returns DataFrame with columns: [frame_id, person_id, x, y]
    """
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=["frame_id", "person_id", "x", "y"]
    )
    df = df.dropna()
    df["frame_id"]  = df["frame_id"].astype(int)
    df["person_id"] = df["person_id"].astype(int)
    df["x"]         = df["x"].astype(float)
    df["y"]         = df["y"].astype(float)
    df = df.sort_values(["person_id", "frame_id"]).reset_index(drop=True)
    logger.info(f"Loaded {filepath} | {len(df)} rows | {df['person_id'].nunique()} pedestrians")
    return df


def extract_trajectories(
    df: pd.DataFrame,
    min_length: int = 8
) -> Dict[int, np.ndarray]:
    """
    Extract per-person trajectory arrays from a loaded DataFrame.

    Returns:
        dict mapping person_id → np.ndarray of shape (T, 2) [x, y]
    """
    trajectories = {}
    for pid, group in df.groupby("person_id"):
        coords = group[["x", "y"]].values  # shape (T, 2)
        if len(coords) >= min_length:
            trajectories[pid] = coords
    logger.info(f"Extracted {len(trajectories)} valid trajectories (min_len={min_length})")
    return trajectories


# ── Sliding window sequence builder ──────────────────────────────────────────
def build_sequences(
    trajectories: Dict[int, np.ndarray],
    obs_len: int = 8,
    pred_len: int = 12,
    skip: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (observed, future) sequence pairs using a sliding window.

    Args:
        trajectories: output of extract_trajectories()
        obs_len:  number of observed timesteps (input to model)
        pred_len: number of future timesteps (target output)
        skip:     stride between frames

    Returns:
        obs   — shape (N, obs_len,  2)  observed positions
        pred  — shape (N, pred_len, 2)  future positions to predict
    """
    total_len = obs_len + pred_len
    obs_list, pred_list = [], []

    for pid, coords in trajectories.items():
        T = len(coords)
        # Slide window over trajectory
        for start in range(0, T - total_len * skip + 1, skip):
            indices = list(range(start, start + total_len * skip, skip))
            if indices[-1] >= T:
                break
            window = coords[indices]          # (total_len, 2)
            obs_list.append(window[:obs_len])
            pred_list.append(window[obs_len:])

    if not obs_list:
        logger.warning("No sequences built — check trajectory lengths vs obs+pred len")
        return np.array([]), np.array([])

    obs  = np.stack(obs_list,  axis=0).astype(np.float32)   # (N, obs_len, 2)
    pred = np.stack(pred_list, axis=0).astype(np.float32)   # (N, pred_len, 2)
    logger.info(f"Built {len(obs)} sequences | obs={obs_len} pred={pred_len}")
    return obs, pred


# ── Normalization ─────────────────────────────────────────────────────────────
def normalize_sequences(
    obs: np.ndarray,
    pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize by subtracting the last observed position (relative coords).
    The model learns displacements, not absolute positions — generalizes better.

    Returns: obs_norm, pred_norm, mean, std
    """
    # Origin = last observed point per sequence
    origin = obs[:, -1:, :]           # (N, 1, 2)
    obs_norm  = obs  - origin
    pred_norm = pred - origin

    mean = obs_norm.mean(axis=(0, 1), keepdims=True)
    std  = obs_norm.std(axis=(0, 1),  keepdims=True) + 1e-8

    obs_norm  = (obs_norm  - mean) / std
    pred_norm = (pred_norm - mean) / std

    return obs_norm, pred_norm, mean, std


# ── PyTorch Dataset ───────────────────────────────────────────────────────────
class ETHUCYDataset(Dataset):
    """
    PyTorch Dataset for ETH/UCY trajectory data.

    Usage:
        dataset = ETHUCYDataset("data/raw/eth_ucy/eth/crowds_zara01.txt")
        loader  = DataLoader(dataset, batch_size=64, shuffle=True)
        obs, target = next(iter(loader))
        # obs:    (batch, obs_len, 2)
        # target: (batch, pred_len, 2)
    """

    def __init__(
        self,
        filepath: str,
        obs_len: int  = 8,
        pred_len: int = 12,
        skip: int = 1,
        min_length: int = 8,
        normalize: bool = True,
    ):
        self.obs_len  = obs_len
        self.pred_len = pred_len
        self.normalize = normalize

        df = load_eth_ucy_file(filepath)
        trajectories = extract_trajectories(df, min_length=min_length)
        obs, pred = build_sequences(trajectories, obs_len, pred_len, skip)

        if normalize and len(obs) > 0:
            obs, pred, self.mean, self.std = normalize_sequences(obs, pred)
        else:
            self.mean = np.zeros((1, 1, 2), dtype=np.float32)
            self.std  = np.ones((1, 1, 2),  dtype=np.float32)

        self.obs  = torch.tensor(obs,  dtype=torch.float32)
        self.pred = torch.tensor(pred, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.obs[idx], self.pred[idx]


# ── Multi-file loader (combine all subsets) ───────────────────────────────────
def load_all_subsets(
    data_dir: str,
    obs_len: int  = 8,
    pred_len: int = 12,
    skip: int = 1,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and combine all ETH/UCY .txt files from a directory.

    Args:
        data_dir: folder containing ETH/UCY annotation .txt files

    Returns:
        Combined obs and pred arrays across all subsets
    """
    all_obs, all_pred = [], []
    txt_files = list(Path(data_dir).rglob("*.txt"))

    if not txt_files:
        logger.warning(f"No .txt files found in {data_dir}")
        return np.array([]), np.array([])

    for fpath in txt_files:
        try:
            df = load_eth_ucy_file(str(fpath))
            trajectories = extract_trajectories(df)
            obs, pred = build_sequences(trajectories, obs_len, pred_len, skip)
            if len(obs) > 0:
                all_obs.append(obs)
                all_pred.append(pred)
        except Exception as e:
            logger.warning(f"Skipping {fpath.name}: {e}")

    if not all_obs:
        return np.array([]), np.array([])

    combined_obs  = np.concatenate(all_obs,  axis=0)
    combined_pred = np.concatenate(all_pred, axis=0)
    logger.info(f"Combined: {len(combined_obs)} total sequences from {len(all_obs)} files")
    return combined_obs, combined_pred


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python eth_ucy_loader.py <path_to_txt_file>")
        print("Example: python eth_ucy_loader.py data/raw/eth_ucy/zara01.txt")
        sys.exit(1)

    filepath = sys.argv[1]
    dataset  = ETHUCYDataset(filepath)
    print(f"\n✅ Dataset loaded: {len(dataset)} sequences")
    obs, pred = dataset[0]
    print(f"   obs shape:  {obs.shape}  (observed trajectory)")
    print(f"   pred shape: {pred.shape} (future to predict)")
    print(f"\nSample observed positions:\n{obs}")
    print(f"\nSample future positions:\n{pred}")