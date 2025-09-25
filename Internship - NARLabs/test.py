import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataloader import default_collate
import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils import *
from Base import Base
from GRU import GRUtorch
from xGRU import xGRU
from LSTM import LSTMtorch
from xLSTM import xLSTM

DAYS_PLOT = 1
ROAD = "m20317"
device = "cuda" if torch.cuda.is_available() else "cpu"


def LoadAndImput(road):
    data_obs = pd.read_csv(f"data/test/obs_{road}.csv", index_col=False)
    data_pred = pd.read_csv(f"data/test/pred_{road}.csv", index_col=False)
    data_obs = data_obs[data_obs["RoadID"] == "m20317"]
    data_pred = data_pred[data_pred["RoadID"] == "m20317"]

    # Drop duplicates and sorted by time and drop columns id and RoadID
    for data, colTime in zip([data_obs, data_pred], ["UpdateTime", "PredictionTime"]):
        data.drop_duplicates(subset=colTime, keep="first", inplace=True)
        data.loc[:, colTime] = pd.to_datetime(data.loc[:, colTime])
        data.sort_values(by=colTime, inplace=True)
        data.drop(columns=["id", "RoadID"], inplace=True)

    # Cut the data the start and finish at the same time and place time as index
    merge = pd.merge(
        data_obs,
        data_pred,
        left_on="UpdateTime",
        right_on="PredictionTime",
        how="inner",
    )
    start_time, end_time = merge["UpdateTime"].min(), merge["UpdateTime"].max()
    date_range = pd.date_range(start=start_time, end=end_time, freq="5min")
    del merge
    for data, colTime in zip([data_obs, data_pred], ["UpdateTime", "PredictionTime"]):
        data = data.loc[(data[colTime] >= start_time) & (data[colTime] <= end_time)]
        data.set_index(colTime, inplace=True)
        data = data.reindex(date_range)
        data.index.name = "time"
        if colTime == "UpdateTime":
            data_obs = data
        else:
            data_pred = data

    # Imputation of missing values
    data_obs = DataProcessing(data_obs).imput()
    data_pred = DataProcessing(data_pred).imput()

    return data_obs, data_pred


def getBatches(data, seq_len=12, shift=4):
    test = data.copy()
    with open(f"models/data_imputed_m20317.pkl", "rb") as file:
        mn, std = pickle.load(file)

    for c in test.columns:
        mean = mn[c]
        stdev = std[c]

        test[c] = (test[c] - mean) / stdev

    test = SequenceDataset(test, seq_len, shift)

    test = DataLoader(test, batch_size=1, shuffle=False)

    return test, (mn, std)


def LoadModels(path="results/best_params/"):
    models = nn.ModuleList(
        [
            Base(LSTMtorch, 3, 300, num_layers=2).to(device),
            Base(GRUtorch, 3, 300, num_layers=2).to(device),
            Base(xLSTM, 3, 300, num_layers=2, signature=(1, 1)).to(device),
            Base(xLSTM, 3, 300, num_layers=2, signature=(1, 0)).to(device),
            Base(xLSTM, 3, 300, num_layers=2, signature=(0, 1)).to(device),
            Base(xGRU, 3, 300, num_layers=2, signature=(1, 1)).to(device),
            Base(xGRU, 3, 300, num_layers=2, signature=(1, 0)).to(device),
            Base(xGRU, 3, 300, num_layers=2, signature=(0, 1)).to(device),
        ]
    )
    loads = [
        torch.load(path + f"{model.__type__}.pth", weights_only=True)
        for model in models
    ]
    for load in loads:
        load_updated = {
            key.replace("_orig_mod.", ""): value for key, value in load.items()
        }
        load.clear()
        load.update(load_updated)

    [model.load_state_dict(load) for model, load in zip(models, loads)]
    return models


def ModelsPrediction(models, test, mn, std):
    dic = {}
    features = list(mn.keys())
    with torch.no_grad():
        for model in models:
            pred = np.array([model.predict(X)[-1, :].tolist() for X, y in test])
            for i, ft in enumerate(features):
                pred[:, i] = pred[:, i] * std[ft] + mn[ft]
            dic[model.__type__] = pred
    return dic


def RMSE(pred):
    target = pred["Raw"]
    dic = {}
    for model, mtx_pred in pred.items():
        rmse = np.sqrt(np.mean((mtx_pred - target) ** 2))
        dic[model] = rmse
    return dic


def Plots(dict, features, rmse, length=40, path="./results/best_params/"):
    ft = len(features)
    fig, axs = plt.subplots(ft, 1, figsize=(20, 10), sharex=True)
    handles = {model: [] for model in dict.keys()}

    for i in range(ft):
        for model, pred in dict.items():
            (handle,) = axs[i].plot(
                pred[-length:, i], label=f"{model} ♦ RMSE: {rmse[model]:.2f}"
            )
            handles[model].append(handle)
        axs[i].set_xlabel("Time")
        axs[i].set_ylabel(features[i])

    # Create legend handles and labels
    legend_handles = [handles[model][0] for model in handles]
    legend_labels = [f"{model} ♦ RMSE: {rmse[model]:.2f}" for model in handles]

    fig.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncol=1,
        title="Models \n \n",
        title_fontsize="13",
        fontsize="12",
        handletextpad=1,
        handlelength=2,
        borderpad=2,
        labelspacing=2,
        fancybox=True,
        framealpha=0,
    )
    plt.figtext(
        1.0855,
        0.74,
        "(average RMSE on the total dataset)",
        ha="center",
        fontsize=12,
    )
    fig.suptitle(f"Predictions plot on the last {length} data points", fontsize=18)
    plt.tight_layout()
    plt.savefig(path + "predictions/test.png", bbox_inches="tight")
    plt.close()


data_obs, data_pred = LoadAndImput(ROAD)
test, (mn, std) = getBatches(data_obs)
models = LoadModels()
pred = ModelsPrediction(models, test, mn, std)
pred["Raw"] = data_obs.to_numpy()
pred["Prediction"] = data_pred.to_numpy()
rmse = RMSE(pred)
Plots(pred, list(mn.keys()), rmse)
