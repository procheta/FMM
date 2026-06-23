# ============================================================
# Autoregressive inference over EVERY possible start sample
# for the fine-tuned ERA5 adapter model
# ============================================================

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from Prithvi_ERA5_Finetune import (
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
SAVE_PATH="/home/ubuntu/FMM/prithvi_wxc_era5_direct_t2m.pt"
LEAD_STEPS=1
def _make_batch(sample, device):
    """Convert one dataset sample to a batch of size 1."""
    return {
        k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else v
        for k, v in sample.items()
    }


def _relative_l2(pred, target, eps=1e-12):
    """Relative L2 error for one 2D field."""
    num = torch.linalg.vector_norm(pred - target)
    den = torch.linalg.vector_norm(target) + eps
    return (num / den).item()


def _plot_error_summary(
    rollout_steps,
    rel_l2_mean,
    rel_l2_std,
    mae_mean,
    mae_std,
    output_path,
):
    """
    Plot mean autoregressive error over all rollout starts.
    The shaded region is +/- one standard deviation across start samples.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Autoregressive Time Evolution Stability Analysis", fontsize=14)

    # Relative L2
    axes[0].plot(rollout_steps, rel_l2_mean, marker="o", label=f"{TARGET_VAR} mean")
    axes[0].fill_between(
        rollout_steps,
        np.maximum(rel_l2_mean - rel_l2_std, 1e-20),
        rel_l2_mean + rel_l2_std,
        alpha=0.2,
        label=r"$\pm$1 std over starts",
    )
    axes[0].set_title("Relative L2 Error vs Rollout Step")
    axes[0].set_xlabel("Autoregressive Rollout Step")
    axes[0].set_ylabel("Relative L2 Error")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[0].legend()

    # Physical-scale MAE
    axes[1].plot(rollout_steps, mae_mean, marker="o", label=f"{TARGET_VAR} mean")
    axes[1].fill_between(
        rollout_steps,
        np.maximum(mae_mean - mae_std, 1e-20),
        mae_mean + mae_std,
        alpha=0.2,
        label=r"$\pm$1 std over starts",
    )
    axes[1].set_title("Mean Absolute Error vs Rollout Step (Physical Scale)")
    axes[1].set_xlabel("Autoregressive Rollout Step")
    axes[1].set_ylabel("Mean Absolute Error")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved autoregressive summary plot to: {output_path}")


@torch.no_grad()
def run_autoregressive_inference_all_starts(
    n_unroll_steps=20,
    test_time_range=("2023-01-05", "2023-05-05"),
    first_start_sample_index=0,
    last_start_sample_index=None,
    max_start_samples=None,
    output_plot="autoregressive_stability_all_starts.png",
    output_csv="autoregressive_stability_all_starts.csv",
):
    """
    Run a 20-step autoregressive rollout starting from EACH possible sample
    in the selected test data range.

    Example:
        If the test dataset has 100 samples and n_unroll_steps=20,
        this evaluates rollouts starting at samples:
            0, 1, 2, ..., 80
        because each start needs 20 future samples.

    At rollout step 0:
        model receives the true ERA5 input for that start sample.

    At rollout steps 1..n_unroll_steps-1:
        the TARGET_VAR input channel is replaced by the previous model prediction.

    This requires TARGET_VAR to be included in INPUT_VARS.
    """
    print("Device:", DEVICE)
    print("Target variable:", TARGET_VAR)
    print("Input variables:", INPUT_VARS)
    print("Lead steps:", LEAD_STEPS)

    if TARGET_VAR not in INPUT_VARS:
        raise ValueError(
            f"TARGET_VAR={TARGET_VAR!r} is not in INPUT_VARS={INPUT_VARS}. "
            "For autoregressive rollout, the predicted target must be one of the input channels."
        )

    target_channel = list(INPUT_VARS).index(TARGET_VAR)

    # -----------------------------
    # 1. Load ERA5 data
    # -----------------------------
    ds = open_era5_zarr(ERA5_DIR, time_range=test_time_range)
    start = pd.Timestamp(test_time_range[0])
    end = pd.Timestamp(test_time_range[1])
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

    if len(dataset) < n_unroll_steps:
        raise ValueError(
            f"Dataset length {len(dataset)} is smaller than n_unroll_steps={n_unroll_steps}."
        )

    max_possible_start = len(dataset) - n_unroll_steps

    if last_start_sample_index is None:
        last_start_sample_index = max_possible_start
    else:
        last_start_sample_index = min(last_start_sample_index, max_possible_start)

    if first_start_sample_index < 0:
        raise ValueError("first_start_sample_index must be >= 0.")

    if first_start_sample_index > last_start_sample_index:
        raise ValueError(
            f"Invalid start range: first_start_sample_index={first_start_sample_index}, "
            f"last_start_sample_index={last_start_sample_index}."
        )

    start_indices = list(range(first_start_sample_index, last_start_sample_index + 1))

    if max_start_samples is not None:
        start_indices = start_indices[:max_start_samples]

    print(f"Dataset length: {len(dataset)}")
    print(f"Rollout length per start: {n_unroll_steps}")
    print(f"Number of rollout starts: {len(start_indices)}")
    print(f"First start index: {start_indices[0]}")
    print(f"Last start index: {start_indices[-1]}")

    # -----------------------------
    # 3. Rebuild and load model
    # -----------------------------
    model = ERA5PrithviWxCFinetuner(
        prithvi_wxc_model=None,
        in_channels=len(INPUT_VARS),
    ).to(DEVICE)

    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from: {SAVE_PATH}")
    print("Checkpoint best val MSE:", checkpoint.get("best_val_mse", "not saved"))

    target_mean = dataset.stats[TARGET_VAR]["mean"]
    target_std = dataset.stats[TARGET_VAR]["std"]

    # errors[start_id, rollout_step]
    rel_l2_all = np.zeros((len(start_indices), n_unroll_steps), dtype=np.float64)
    mae_phys_all = np.zeros((len(start_indices), n_unroll_steps), dtype=np.float64)

    # -----------------------------
    # 4. Nested autoregressive evaluation
    # -----------------------------
    for start_id, start_sample_index in enumerate(start_indices):
        previous_prediction = None

        for rollout_step in range(n_unroll_steps):
            sample_index = start_sample_index + rollout_step
            sample = dataset[sample_index]
            batch = _make_batch(sample, DEVICE)


            # Autoregressive feedback:
            # For step > 0, replace the true TARGET_VAR channel at the current time
            # with the model prediction from the previous rollout step.
            if rollout_step > 0:
                batch["x"][:, target_channel:target_channel + 1, :, :] = previous_prediction

            pred = model(batch)       # [1, 1, H, W], normalized
            target = batch["target"]  # [1, 1, H, W], normalized

            rel_l2_all[start_id, rollout_step] = _relative_l2(pred, target)

            pred_phys = pred * target_std + target_mean
            target_phys = target * target_std + target_mean
            mae_phys_all[start_id, rollout_step] = F.l1_loss(pred_phys, target_phys).item()

            previous_prediction = pred.detach()

        if (start_id + 1) % 10 == 0 or (start_id + 1) == len(start_indices):
            print(
                f"Completed {start_id + 1}/{len(start_indices)} rollouts "
                f"(latest start={start_sample_index})"
            )

    # -----------------------------
    # 5. Aggregate over all start samples
    # -----------------------------
    rollout_steps = np.arange(1, n_unroll_steps + 1)

    rel_l2_mean = rel_l2_all.mean(axis=0)
    rel_l2_std = rel_l2_all.std(axis=0)

    mae_mean = mae_phys_all.mean(axis=0)
    mae_std = mae_phys_all.std(axis=0)

    df = pd.DataFrame({
        "rollout_step": rollout_steps,
        "relative_l2_mean": rel_l2_mean,
        "relative_l2_std": rel_l2_std,
        "mae_physical_mean": mae_mean,
        "mae_physical_std": mae_std,
    })
    df.to_csv(output_csv, index=False)
    print(f"Saved autoregressive summary CSV to: {output_csv}")

    _plot_error_summary(
        rollout_steps=rollout_steps,
        rel_l2_mean=rel_l2_mean,
        rel_l2_std=rel_l2_std,
        mae_mean=mae_mean,
        mae_std=mae_std,
        output_path=output_plot,
    )

    return {
        "start_indices": np.array(start_indices),
        "rollout_steps": rollout_steps,
        "relative_l2_all": rel_l2_all,
        "mae_physical_all": mae_phys_all,
        "relative_l2_mean": rel_l2_mean,
        "relative_l2_std": rel_l2_std,
        "mae_physical_mean": mae_mean,
        "mae_physical_std": mae_std,
    }


if __name__ == "__main__":
    results = run_autoregressive_inference_all_starts(
        n_unroll_steps=5,
        test_time_range=("2020-01-05", "2020-01-15"),

        # Use every possible start by default.
        first_start_sample_index=0,
        last_start_sample_index=None,

        # Set this to a small number, e.g. 10, for quick debugging.
        max_start_samples=None,

        output_plot="autoregressive_stability_all_starts.png",
        output_csv="autoregressive_stability_all_starts.csv",
    )
