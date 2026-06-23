# ============================================================
# Inference script for the fine-tuned ERA5 adapter model
# ============================================================

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

SAVE_PATH="prithvi_wxc_era5_direct_t2m.pt"

# Import from your training script.
# Rename your training file to something importable, e.g. prithvi_test_fixed.py
from prithvi_test_fixed import (
    ERA5_DIR,
    DEVICE,
    INPUT_VARS,
    TARGET_VAR,
    LEAD_STEPS,
    SELECTED_LEVEL,
    SAVE_PATH,
    open_era5_zarr,
    ERA5DirectForecastDataset,
    ERA5PrithviWxCFinetuner,
)


@torch.no_grad()
def run_inference(sample_index=0):
    print("Device:", DEVICE)

    # -----------------------------
    # 1. Load ERA5 data
    # -----------------------------



    TEST_TIME_RANGE = ("2023-01-05", "2023-05-05")

    ds = open_era5_zarr(ERA5_DIR,time_range=TEST_TIME_RANGE)

    # Slice only the test period
    start = pd.Timestamp(TEST_TIME_RANGE[0])
    end = pd.Timestamp(TEST_TIME_RANGE[1])
    ds = ds.sel(time=slice(start, end))

    print("Testing on time range:", ds.time.values[0], "to", ds.time.values[-1])
    print("Number of test timesteps:", ds.sizes["time"])

    task_vars = sorted(set(INPUT_VARS + [TARGET_VAR]))
    ds = ds[task_vars]

    print("Loading selected ERA5 test subset into memory...")
    ds = ds.load()



    # -----------------------------
    # 2. Rebuild dataset
    # -----------------------------
    dataset = ERA5DirectForecastDataset(
        ds=ds,
        input_vars=INPUT_VARS,
        target_var=TARGET_VAR,
        lead_steps=LEAD_STEPS,
        selected_level=SELECTED_LEVEL,
        normalize=True,
    )

    if sample_index >= len(dataset):
        raise IndexError(
            f"sample_index={sample_index} is out of range. "
            f"Dataset length is {len(dataset)}."
        )

    # Single sample
    sample = dataset[sample_index]

    batch = {
        k: v.unsqueeze(0).to(DEVICE) if torch.is_tensor(v) else v
        for k, v in sample.items()
    }

    # -----------------------------
    # 3. Rebuild model
    # -----------------------------
    model = ERA5PrithviWxCFinetuner(
        prithvi_wxc_model=None,
        in_channels=len(INPUT_VARS),
    ).to(DEVICE)

    # -----------------------------
    # 4. Load checkpoint
    # -----------------------------
    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from: {SAVE_PATH}")
    print("Checkpoint best val MSE:", checkpoint.get("best_val_mse", "not saved"))

    # -----------------------------
    # 5. Predict
    # -----------------------------
    pred = model(batch)              # [1, 1, H, W]
    target = batch["target"]         # [1, 1, H, W]

    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()

    print(f"Sample index: {sample_index}")
    print(f"Prediction shape: {tuple(pred.shape)}")
    print(f"Target shape: {tuple(target.shape)}")
    print(f"Normalized MSE: {mse:.6f}")
    print(f"Normalized MAE: {mae:.6f}")

    # -----------------------------
    # 6. Denormalize prediction
    # -----------------------------
    target_mean = dataset.stats[TARGET_VAR]["mean"]
    target_std = dataset.stats[TARGET_VAR]["std"]

    pred_denorm = pred.cpu().squeeze().numpy() * target_std + target_mean
    target_denorm = target.cpu().squeeze().numpy() * target_std + target_mean

    mse_denorm = np.mean((pred_denorm - target_denorm) ** 2)
    mae_denorm = np.mean(np.abs(pred_denorm - target_denorm))

    print(f"Denormalized MSE: {mse_denorm:.6f}")
    print(f"Denormalized MAE: {mae_denorm:.6f}")

    return pred_denorm, target_denorm


if __name__ == "__main__":
    pred, target = run_inference(sample_index=0)
