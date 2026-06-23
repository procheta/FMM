# ============================================================
# Fine-tune a small ERA5 adapter directly on ERA5 Zarr data
# No ERA5 -> MERRA-2 conversion.
#
# Note: this script verifies the ERA5/Zarr training pipeline and uses a
# lightweight adapter head. It does NOT yet call the full Prithvi-WxC backbone.
# ============================================================

import random
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


# -----------------------------
# 1. Config
# -----------------------------

ERA5_DIR = Path("/home/ubuntu/era5")   # folder containing .zarr stores
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 1
NUM_EPOCHS = 10
LR = 1e-5
WEIGHT_DECAY = 1e-4

LEAD_STEPS = 1          # predict t+1
SELECTED_LEVEL = 0      # index of pressure level to use
SAVE_PATH = "prithvi_wxc_era5_direct_t2m.pt"

# Your current ERA5 Zarr data contains: q, t, u, v, w, z
INPUT_VARS = ["t", "u", "v", "q", "z"]
TARGET_VAR = "t"

# Use None to train over the full available time range.
# Example subset: ("2020-01-01", "2020-01-03")
TIME_RANGE = ("2020-01-01", "2022-01-01")
# TIME_RANGE = None

# -----------------------------
# 2. Utility functions
# -----------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    """Standardize common ERA5 coordinate names."""
    rename = {}

    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    if "valid_time" in ds.coords:
        rename["valid_time"] = "time"

    if rename:
        ds = ds.rename(rename)

    if "lon" in ds.coords and float(ds.lon.max()) > 180:
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
        ds = ds.sortby("lon")

    if "lat" in ds.coords:
        ds = ds.sortby("lat")

    return ds


def open_era5_zarr(era5_dir: Path, time_range=None) -> xr.Dataset:
    """
    Open ERA5 Zarr stores. Handles stores where each file has a scalar
    time coordinate instead of a real time dimension.
    """
    zarr_paths = sorted(list(era5_dir.rglob("*.zarr")))

    if len(zarr_paths) == 0:
        raise FileNotFoundError(f"No .zarr stores found under {era5_dir}")

    print(f"Found {len(zarr_paths)} Zarr stores.")
    for p in zarr_paths[:10]:
        print(" ", p)

    datasets = []

    for p in zarr_paths:
        try:
            ds_i = xr.open_zarr(p, consolidated=True)
        except Exception:
            ds_i = xr.open_zarr(p, consolidated=False)

        ds_i = standardize_coords(ds_i)

        # time is a scalar coordinate; convert it into a dimension
        if "time" in ds_i.coords and "time" not in ds_i.dims:
            time_value = pd.to_datetime(ds_i["time"].values)
            ds_i = ds_i.drop_vars("time")
            ds_i = ds_i.expand_dims(time=[time_value])

        # valid_time is a scalar coordinate; convert it into a dimension
        elif "valid_time" in ds_i.coords and "time" not in ds_i.dims:
            time_value = pd.to_datetime(ds_i["valid_time"].values)
            ds_i = ds_i.drop_vars("valid_time")
            ds_i = ds_i.expand_dims(time=[time_value])

        # time is already a dimension
        elif "time" in ds_i.dims:
            ds_i["time"] = pd.to_datetime(ds_i["time"].values)

        else:
            raise ValueError(
                f"No usable time coordinate found in {p}. "
                f"dims={ds_i.dims}, coords={list(ds_i.coords)}, vars={list(ds_i.data_vars)}"
            )

        datasets.append(ds_i)

    ds = xr.concat(
        datasets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )

    ds = ds.sortby("time")

    if time_range is not None:
        start = pd.Timestamp(time_range[0])
        end = pd.Timestamp(time_range[1])
        ds = ds.sel(time=slice(start, end))
    else:
        print("Using full available time range:", ds.time.values[0], "to", ds.time.values[-1])

    if ds.sizes.get("time", 0) == 0:
        raise ValueError(f"No timesteps found in TIME_RANGE={TIME_RANGE}.")

    print(ds)
    print("Variables:", list(ds.data_vars))
    print("Dims:", ds.dims)
    print("Time range:", ds.time.values[0], "to", ds.time.values[-1])

    return ds


# -----------------------------
# 3. Direct ERA5 Dataset
# -----------------------------

class ERA5DirectForecastDataset(Dataset):
    """Direct ERA5 forecasting dataset."""

    def __init__(
        self,
        ds: xr.Dataset,
        input_vars: list[str],
        target_var: str,
        lead_steps: int = 1,
        selected_level: int = 0,
        normalize: bool = True,
    ):
        self.ds = ds
        self.input_vars = input_vars
        self.target_var = target_var
        self.lead_steps = lead_steps
        self.selected_level = selected_level
        self.normalize = normalize

        missing = [v for v in set(input_vars + [target_var]) if v not in ds.data_vars]
        if missing:
            raise KeyError(
                f"Missing variables: {missing}. Available variables: {list(ds.data_vars)}"
            )

        self.time_len = ds.sizes["time"]
        if self.time_len <= self.lead_steps:
            raise ValueError(
                f"Need more timesteps than lead_steps. Got time={self.time_len}, "
                f"lead_steps={self.lead_steps}."
            )

        self.lat = ds["lat"].values
        self.lon = ds["lon"].values

        lat_r = torch.as_tensor(self.lat / 360.0 * 2.0 * np.pi, dtype=torch.float32)
        lon_r = torch.as_tensor(self.lon / 360.0 * 2.0 * np.pi, dtype=torch.float32)
        lat_grid, lon_grid = torch.meshgrid(lat_r, lon_r, indexing="ij")

        self.static = torch.stack(
            [torch.sin(lat_grid), torch.cos(lon_grid), torch.sin(lon_grid)],
            dim=0,
        )

        self.stats = {}
        if self.normalize:
            for v in set(input_vars + [target_var]):
                da = self.ds[v]
                if "level" in da.dims:
                    da = da.isel(level=self.selected_level)

                # If ds is loaded into memory, .values is immediate.
                self.stats[v] = {
                    "mean": float(np.asarray(da.mean().values)),
                    "std": float(np.asarray(da.std().values)) + 1e-6,
                }

        print("Dataset statistics:")
        for k, val in self.stats.items():
            print(f"  {k}: mean={val['mean']:.6f}, std={val['std']:.6f}")

    def __len__(self):
        return self.time_len - self.lead_steps

    def _load_2d(self, var_name: str, time_index: int) -> torch.Tensor:
        da = self.ds[var_name].isel(time=time_index)

        if "level" in da.dims:
            da = da.isel(level=self.selected_level)

        da = da.transpose("lat", "lon")
        arr = np.asarray(da.values)
        tensor = torch.as_tensor(arr, dtype=torch.float32)

        if self.normalize:
            mean = self.stats[var_name]["mean"]
            std = self.stats[var_name]["std"]
            tensor = (tensor - mean) / std

        return tensor

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        x_vars = [self._load_2d(v, index) for v in self.input_vars]

        # [C, H, W]
        x = torch.stack(x_vars, dim=0)

        # [T, C, H, W], here T=1
        x = x.unsqueeze(0)

        # [1, H, W]
        target = self._load_2d(self.target_var, index + self.lead_steps).unsqueeze(0)

        return {
            "x": x,
            "y": x.squeeze(0),
            "target": target,
            "lead_time": torch.tensor([self.lead_steps], dtype=torch.float32),
            "static": self.static,
        }


# -----------------------------
# 4. Adapter model
# -----------------------------

class ERA5PrithviWxCFinetuner(nn.Module):
    """
    Lightweight ERA5 adapter for testing the data/training pipeline.
    This is not the full Prithvi-WxC backbone.
    """

    def __init__(self, prithvi_wxc_model: nn.Module | None, in_channels: int):
        super().__init__()
        self.prithvi = prithvi_wxc_model

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels + 3, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.t2m_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, batch):
        x = batch["x"][:, -1]       # [B, C, H, W]
        static = batch["static"]    # [B, 3, H, W]

        x = torch.cat([x, static], dim=1)
        h = self.input_proj(x)
        pred = self.t2m_head(h)
        return pred


# -----------------------------
# 5. Training functions
# -----------------------------

def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = move_batch_to_device(batch, DEVICE)
        optimizer.zero_grad(set_to_none=True)

        pred = model(batch)
        target = batch["target"]
        loss = F.mse_loss(pred, target)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Validation", leave=False):
        batch = move_batch_to_device(batch, DEVICE)
        pred = model(batch)
        target = batch["target"]
        loss = F.mse_loss(pred, target)
        total_loss += loss.item()

    return total_loss / max(1, len(loader))


# -----------------------------
# 6. Main
# -----------------------------

def main():
    set_seed(42)
    print("Device:", DEVICE)

    print("Opening ERA5 Zarr data directly...")
    ds = open_era5_zarr(ERA5_DIR)

    # Keep only the variables needed for the task.
    task_vars = sorted(set(INPUT_VARS + [TARGET_VAR]))
    ds = ds[task_vars]

    # Load the small selected subset into memory to avoid slow dask/zarr reads per batch.
    print("Loading selected ERA5 subset into memory...")
    ds = ds.load()
    print("Loaded selected ERA5 subset into memory.")

    dataset = ERA5DirectForecastDataset(
        ds=ds,
        input_vars=INPUT_VARS,
        target_var=TARGET_VAR,
        lead_steps=LEAD_STEPS,
        selected_level=SELECTED_LEVEL,
        normalize=True,
    )

    n = len(dataset)
    train_size = max(1, int(0.8 * n))
    val_size = n - train_size

    # Ensure non-empty validation if possible.
    if val_size == 0 and n > 1:
        train_size = n - 1
        val_size = 1

    print(f"Dataset samples: {n}, train: {train_size}, val: {val_size}")

    if val_size > 0:
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train_ds = dataset
        val_ds = None

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=10,
        pin_memory=False,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=10,
            pin_memory=False,
        )

    # Quick sanity check before training.
    batch = next(iter(train_loader))
    print("Sanity-check batch shapes:")
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)}")

    prithvi = None
    model = ERA5PrithviWxCFinetuner(
        prithvi_wxc_model=prithvi,
        in_channels=len(INPUT_VARS),
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer)

        if val_loader is not None:
            val_loss = evaluate(model, val_loader)
        else:
            val_loss = train_loss

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_mse": best_val,
                    "input_vars": INPUT_VARS,
                    "target_var": TARGET_VAR,
                    "lead_steps": LEAD_STEPS,
                    "selected_level": SELECTED_LEVEL,
                },
                SAVE_PATH,
            )
            print(f"Saved best checkpoint to: {SAVE_PATH}")

    print("Done.")


if __name__ == "__main__":
    main()
