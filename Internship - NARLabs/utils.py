import os
import warnings
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset


from einops import rearrange
from torch import Tensor
from typing import List
from statsmodels.tsa.seasonal import STL

device = "cuda" if torch.cuda.is_available() else "cpu"


class SequenceDataset(Dataset):
    def __init__(self, dataframe, seq_len, shift):
        self.device = device
        self.seq_len = seq_len
        self.shift = shift
        self.y = torch.tensor(dataframe.values).float().to(device)
        self.X = torch.tensor(dataframe.values).float().to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        # if the i is on the beginning
        if i >= self.seq_len - 1:
            i_start = i - self.seq_len + 1
            x = self.X[i_start : (i + 1), :]
        else:
            padding = self.X[0].repeat(self.seq_len - i - 1, 1)
            x = self.X[0 : (i + 1), :]
            x = torch.cat((padding, x), 0)

        # if the i is on the end
        if i <= self.X.size(1) - self.shift:
            y = self.y[i + self.shift]
        else:
            y = x[-1, :]
        return x, y


def train_model(batches, model, criterion, optimizer):
    total_loss = 0
    for X, y in batches:
        loss = model.fit(X, y, criterion, optimizer)
        total_loss += loss

    return total_loss / len(batches)


def test_model(batches, model, criterion):
    total_loss = 0
    for X, y in batches:
        pred = model.predict(X)
        total_loss += criterion(pred, y).item()

    return total_loss / len(batches)


def register(val_losses, models, path="./results/"):
    if not os.path.exists(path):
        os.makedirs(path)
    # Save losses to CSV file
    with open(path + "losses.csv", "w+", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([model.__type__ for model in models])  # header row
        writer.writerows(val_losses)  # write loss values

    # Plot losses and save to PNG file
    plt.figure(figsize=(20, 10))
    colors = plt.get_cmap("tab10")
    for i, model in enumerate(models):
        plt.plot(
            [loss[i] for loss in val_losses],
            label=model.__type__,
            color=colors(i),
        )

    arg = np.argmin(val_losses)
    argmin = (
        arg // val_losses.shape[1],
        arg % val_losses.shape[1],
    )
    plt.axvline(x=argmin[0], color=colors(argmin[1]), linestyle="--", alpha=0.5)

    plt.xlabel("Epochs")
    plt.ylabel("Log Loss")
    plt.yscale("log")
    plt.title("Validation Losses over Epochs (MSE)")
    plt.legend()
    plt.savefig(path + "losses.png", bbox_inches="tight")
    plt.close()

    best_params_dir = path + "best_params/"
    if not os.path.exists(best_params_dir):
        os.makedirs(best_params_dir)

    for i, model in enumerate(models):
        best_epoch = np.argmin(val_losses[:, i])
        if best_epoch == len(val_losses) - 1:  # Check if the last epoch is the best
            torch.save(
                model.state_dict(),
                f"{best_params_dir}{model.__type__}.pth",
            )


def predict_plot(models, val, features, correction, path="./results/best_params/"):
    mn, std = correction
    [
        model.load_state_dict(
            torch.load(path + f"{model.__type__}.pth", weights_only=True)
        )
        for model in models
    ]
    labels = ["row"] + [model.__type__ for model in models]
    row = np.array([y[-1, :].tolist() for X, y in val])
    num_features = row.shape[1]

    fig, axs = plt.subplots(
        num_features, 1, figsize=(20, 3 * num_features), sharex=True
    )
    if num_features == 1:
        axs = [axs]

    feature_pred = []
    fig.suptitle("Prediction on Validation Set", fontsize=20)
    for i, ax in enumerate(axs):
        values = row[:, i] * std[features[i]] + mn[features[i]]
        ax.plot(values, label="values")
        ax.set_ylabel(f"{features[i]}")
        ax.set_xlabel("Time")
        feature_pred.append({"Values": values})

    with torch.no_grad():
        for model in models:
            pred_values = np.array([model.predict(X)[-1, :].tolist() for X, y in val])
            for i in range(num_features):
                pred = pred_values[:, i] * std[features[i]] + mn[features[i]]
                axs[i].plot(
                    pred,
                    label=f"{model.__type__}",
                )
                feature_pred[i][f"{model.__type__}"] = pred

    if not os.path.exists(path + f"predictions"):
        os.makedirs(path + f"predictions")

    for i, fp in enumerate(feature_pred):
        df = pd.DataFrame(fp)
        df.to_csv(path + f"predictions/{features[i]}.csv", index=False)

    fig.legend(labels=labels)
    plt.tight_layout()
    plt.savefig(path + "predictions/val.png", bbox_inches="tight")
    plt.close()


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
    ):
        self._padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self._padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, inp: Tensor) -> Tensor:
        # Handle the case where input has only two dimensions
        # we expect them to have semantics (batch, channels),
        # so we add the missing dimension manually
        if inp.dim() == 2:
            inp = rearrange(inp, "b i -> b 1 i")

        result = super(CausalConv1d, self).forward(inp)
        if self._padding != 0:
            return result[..., : -self._padding]
        return result


class BlockLinear(nn.Module):
    def __init__(
        self,
        block_dims: List[int | List[int]],
        bias: bool = False,
    ):
        super(BlockLinear, self).__init__()

        self._blocks = nn.ParameterList(
            [nn.Parameter(torch.randn(size, requires_grad=True)) for size in block_dims]
        )

        self._bias = nn.Parameter(torch.zeros(sum(block_dims))) if bias else None

    def forward(self, inp: Tensor) -> Tensor:
        full = torch.block_diag(*self._blocks)

        out = torch.matmul(inp, full)

        if self._bias is not None:
            out = out + self._bias

        return out


def enlarge_as(src: Tensor, other: Tensor) -> Tensor:
    """
    Add sufficient number of singleton dimensions
    to tensor a **to the right** so to match the
    shape of tensor b. NOTE that simple broadcasting
    works in the opposite direction.
    """
    return rearrange(src, f'... -> ...{" 1" * (other.dim() - src.dim())}').contiguous()


class DataProcessing:
    def __init__(self, df, plot=False, download=False):
        assert isinstance(df, pd.DataFrame), "Input is not a DataFrame"
        assert isinstance(
            df.index, pd.DatetimeIndex
        ), "Index is not of type DatetimeIndex."
        self.df = df
        self.freq = df.index.freq.n
        self.plot = [plot] if isinstance(plot, str) else plot
        self.dwl = download
        self.week = self.average_week(df)
        self.mask = self.missing()

    def imput(self):
        df_raw = self.df.copy()
        df_hour_imputation = self.hour_imputation(df_raw.copy())
        self.df.update(df_hour_imputation)
        df_week_imputation = self.week_imputation(df_raw.copy())
        self.df.update(df_week_imputation)
        df_month_imputation = self.month_imputation(self.df)
        self.df.update(df_month_imputation)
        if self.df.isna().any().any():
            warnings.warn(
                "The DataFrame still includes NaN after imputation. Try to imput again."
            )
            self.imput()
        return self.df

    def hour_imputation(self, df):
        nan_idx = df.index[df.isna().all(axis=1)]  # get all missing values
        groups = nan_idx.groupby(
            (nan_idx.to_series().diff() != pd.Timedelta(minutes=5)).cumsum()
        )  # group by continue block
        for idx in groups.values():
            if len(idx) <= 60 // self.freq:
                before_idx = idx[0] - pd.Timedelta(minutes=self.freq)
                after_idx = idx[-1] + pd.Timedelta(minutes=self.freq)
                before_row = (
                    df.loc[before_idx] if before_idx in df.index else df.loc[after_idx]
                )  # else if the first data is a NaN
                after_row = (
                    df.loc[after_idx] if after_idx in df.index else df.loc[before_idx]
                )  # else if the last data is a NaN
                values = np.linspace(before_row, after_row, len(idx))
                df.loc[idx] = values
        return df

    def week_imputation(self, df):
        nan_idx = df.index[df.isna().all(axis=1)]  # get all missing values
        groups = nan_idx.groupby(
            (nan_idx.to_series().diff() != pd.Timedelta(minutes=5)).cumsum()
        )  # group by continue block
        for idx in groups.values():
            if (
                60 // self.freq < len(idx) <= 60 * 24 * 7 // self.freq
            ):  # because week imputation
                for i in idx:
                    day, hour, minute = i.day_name(), i.hour, i.minute
                    df.loc[i] = self.week[day].loc[(hour, minute)]
        return df

    def month_imputation(self, df):
        nan_idx = df.index[df.isna().all(axis=1)]
        groups = nan_idx.groupby(
            (nan_idx.to_series().diff() != pd.Timedelta(minutes=5)).cumsum()
        )  # group by continue block
        for idx in groups.values():
            if 60 * 24 * 7 // self.freq < len(idx):
                for column in df.columns:
                    stl = STL(
                        df[column].interpolate(), period=60 * 24 * 7 * 4 // self.freq
                    ).fit()
                    road_deseasonalised_imputed = (
                        df[column] - stl.seasonal
                    ).interpolate(method="linear")
                    road_imputed = road_deseasonalised_imputed + np.random.normal(
                        loc=stl.seasonal, scale=np.std(stl.resid)
                    )
                    df.loc[idx, column] = road_imputed[idx]
        return df

    ### Useful Fonctions ###
    def average_week(self, df):
        daily_average = {}
        daily_groups = df.groupby(df.index.day_name())
        for day, group in daily_groups:
            hourly_groups = (
                group.groupby([group.index.hour, group.index.minute])
                .median()
                .rename_axis(index=["hour", "minute"])
            )
            daily_average[day] = hourly_groups
        return daily_average

    def missing(self):
        len_miss = self.df.isna().all(axis=1).sum()
        len_data = len(self.df)
        ratio = len_miss / len_data
        if len_data < 7 * 24 * 60 // self.freq:
            warnings.warn(
                f"Number of rows under one week : average weekend can not be compute"
            )
        if ratio > 0.1:
            warnings.warn(
                f"Pourcentage of missing values are above 10% : {round(ratio*100, 2)}%. The imputation can introduce biais and vanish the variance."
            )
        print(
            "There are ",
            len_miss,
            " missing values for ",
            len_data,
            " rows, so a ratio of ",
            round(ratio, 2),
        )
        return self.df.isna().all(axis=1).index
