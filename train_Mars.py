import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from pylab import *
from scipy.io import *
import xarray as xr

ds = xr.open_dataset("openmars_my28_ls27_my28_ls41.nc")

print(ds)

lan=ds["lat"]
let=ds["lon"]
tsurf=ds["tsurf"]
ps=ds["ps"]
time=ds["time"]




#sys.exit()
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from terratorch.tasks import PixelwiseRegressionTask
from terratorch.datasets import HLSBands



task = PixelwiseRegressionTask(
    model_args={
        "backbone": "prithvi_eo_v2_300",
        "backbone_pretrained": True,
        "decoder": "FCNDecoder",
        "decoder_channels": 128,
        "num_outputs":4
    },
    model_factory="EncoderDecoderFactory",
    loss="huber",
    optimizer="AdamW",
    lr=1e-1,
)




logger = TensorBoardLogger("tb_logs", name="multivar_regression")
checkpoint = ModelCheckpoint(
    monitor="val/loss",
    mode="min",
    save_top_k=1,
)


from lightning.pytorch.callbacks import Callback

class PrintLossCallback(Callback):

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train/loss")
        val_loss = trainer.callback_metrics.get("val/loss")

        epoch = trainer.current_epoch

        if train_loss is not None:
            print(f"Epoch {epoch} - Train Loss: {train_loss:.6f}")

        if val_loss is not None:
            print(f"Epoch {epoch} - Val Loss: {val_loss:.6f}")




import matplotlib.pyplot as plt
import torch

class PlotPredictionsCallback(Callback):

    def __init__(self, every_n_epochs=5):
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):

        epoch = trainer.current_epoch

        if epoch % self.every_n_epochs != 0:
            return
        val_loader = trainer.datamodule.val_dataloader()
        batch = next(iter(val_loader))

        x = batch["image"].to(pl_module.device)
        y = batch["mask"].cpu()

        with torch.no_grad():
            output = pl_module.model(x)
            pred = output.output.cpu()

        # select first sample in batch
        gt = y[0]
        prediction = pred[0]

        fig, axes = plt.subplots(2, gt.shape[0], figsize=(4 * gt.shape[0], 6))

        for i in range(gt.shape[0]):

            axes[0, i].imshow(gt[i], origin="lower")
            axes[0, i].set_title(f"Ground Truth {i}")
            axes[0, i].axis("off")

            axes[1, i].imshow(prediction[i], origin="lower")
            axes[1, i].set_title(f"Prediction {i}")
            axes[1, i].axis("off")

        plt.tight_layout()

        fig.savefig("epoch_"+str(epoch)+".png")
        plt.close(fig)

trainer = pl.Trainer(
    max_epochs=20,
    logger=logger,
    callbacks=[checkpoint, PrintLossCallback(),PlotPredictionsCallback()]
)



import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from torchgeo.datamodules import NonGeoDataModule


class MarsDataset(Dataset):
    def __init__(
        self,
        nc_path: str,
        input_vars: list[str],
        target_vars: list[str],
        indices: list[int] | None = None,
        forecast_horizon: int = 1,
    ):
        self.ds = xr.open_dataset(nc_path)

        self.input_vars = input_vars
        self.target_vars = target_vars
        self.forecast_horizon = forecast_horizon

        self.inputs = {
            v: self.ds[v].values.astype(np.float32) for v in input_vars
        }
        self.targets = {
            v: self.ds[v].values.astype(np.float32) for v in target_vars
        }

        n_time = next(iter(self.inputs.values())).shape[0]
        max_start = n_time - forecast_horizon

        if indices is None:
            self.indices = list(range(max_start))
        else:
            self.indices = [i for i in indices if i < max_start]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        t_out = t + self.forecast_horizon

        x = np.stack([self.inputs[v][t] for v in self.input_vars], axis=0)   # [C, H, W]
        y = np.stack([self.targets[v][t_out] for v in self.target_vars], axis=0)  # [T, H, W]
        H, W = x.shape[1], x.shape[2]
        dummy1 = np.zeros((1, H, W), dtype=np.float32)
        dummy2 = np.zeros((1, H, W), dtype=np.float32)

        x = np.concatenate([x, dummy1, dummy2], axis=0) 

        sample = {
            "image": torch.tensor(x, dtype=torch.float32),
            "mask": torch.tensor(y, dtype=torch.float32),
        }
        return sample

    def plot(self, sample):
        image = sample["image"]
        mask = sample["mask"]

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        n_in = image.shape[0]
        n_out = mask.shape[0]
        ncols = max(n_in, n_out)

        fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8))
        if ncols == 1:
            axes = np.array(axes).reshape(2, 1)

        for i in range(ncols):
            axes[0, i].axis("off")
            axes[1, i].axis("off")

        for i in range(n_in):
            axes[0, i].imshow(image[i], origin="lower")
            if i < len(self.input_vars):
                title = f"Input {self.input_vars[i]}"
            else:
                title = f"Input dummy_{i - len(self.input_vars) + 1}"

            axes[0, i].set_title(title)
            axes[0, i].axis("on")

        for i in range(n_out):
            axes[1, i].imshow(mask[i], origin="lower")
            if i < len(self.target_vars):
                title = f"Target {self.target_vars[i]}"
            else:
                title = f"Target {i}"

            axes[1, i].set_title(title)
            axes[1, i].axis("on")

        plt.tight_layout()
        return fig


class MarsPixelwiseRegressionDataModule(NonGeoDataModule):
    def __init__(
        self,
        nc_path: str,
        input_vars: list[str],
        target_vars: list[str],
        batch_size: int = 8,
        num_workers: int = 0,
        forecast_horizon: int = 1,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        pin_memory: bool = False,
        drop_last: bool = True,
        **kwargs,
    ):
        super().__init__(MarsDataset, batch_size=batch_size, num_workers=num_workers, **kwargs)

        self.nc_path = nc_path
        self.input_vars = input_vars
        self.target_vars = target_vars
        self.forecast_horizon = forecast_horizon
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def setup(self, stage: str | None = None):
        ds = xr.open_dataset(self.nc_path)
        n_time = ds[self.input_vars[0]].shape[0]
        max_start = n_time - self.forecast_horizon

        all_idx = list(range(max_start))

        n_train = int(self.train_frac * len(all_idx))
        n_val = int(self.val_frac * len(all_idx))

        train_idx = all_idx[:n_train]
        val_idx = all_idx[n_train:n_train + n_val]
        test_idx = all_idx[n_train + n_val:]

        if stage in ("fit", None):
            self.train_dataset = MarsDataset(
                nc_path=self.nc_path,
                input_vars=self.input_vars,
                target_vars=self.target_vars,
                indices=train_idx,
                forecast_horizon=self.forecast_horizon,
            )
            self.val_dataset = MarsDataset(
                nc_path=self.nc_path,
                input_vars=self.input_vars,
                target_vars=self.target_vars,
                indices=val_idx,
                forecast_horizon=self.forecast_horizon,
            )

        if stage in ("validate", None):
            self.val_dataset = MarsDataset(
                nc_path=self.nc_path,
                input_vars=self.input_vars,
                target_vars=self.target_vars,
                indices=val_idx,
                forecast_horizon=self.forecast_horizon,
            )

        if stage in ("test", None):
            self.test_dataset = MarsDataset(
                nc_path=self.nc_path,
                input_vars=self.input_vars,
                target_vars=self.target_vars,
                indices=test_idx,
                forecast_horizon=self.forecast_horizon,
            )

    def _dataloader_factory(self, split: str):
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=(split == "train" and self.drop_last),
            pin_memory=self.pin_memory,
        )





mars_dm = MarsPixelwiseRegressionDataModule(
    nc_path="openmars_my28_ls27_my28_ls41.nc",
    input_vars=["tsurf", "ps","dustcol","co2ice"],      # example inputs
    target_vars=["tsurf", "ps","dustcol","co2ice"],     # example outputs
    batch_size=8,
    num_workers=2,
    forecast_horizon=1,
)
trainer.fit(task, datamodule=mars_dm)






