import pickle
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataloader import default_collate
import time


from utils import *

from Base import Base
from LSTM import LSTMtorch
from GRU import GRUtorch, GRU
from xLSTM import xLSTM
from xGRU import xGRU


torch.manual_seed(420)
##########################
#### Global Variables ####
EPOCHS = 100
BATCH_SIZE = 128
SEQ_LEN = 60 // 5  # one hour length # ! Complexity in O(SEQ_LEN)
HIDDEN_SIZE = 300
LEARNING_RATE = 1e-3

SHIFT = 20 // 5  # for 20min forecast
SPLIT_RATIO = 0.75
FILE_NAME = "data_imputed_m2FS01"
NB_DATA = 2000  # ! Complexity in O(NB_DATA)

device = "cuda" if torch.cuda.is_available() else "cpu"
##########################
##########################

#### Data #####
# import the dataset without the time column
data = pd.read_csv(f"data/{FILE_NAME}.csv", index_col=False)
data = data.drop(data.columns[0], axis=1).tail(NB_DATA).reset_index(drop=True)
split = int(len(data) * SPLIT_RATIO)
train = data.loc[:split].copy()
val = data.loc[split:].copy()

# normalize the dataset
std = {}
mn = {}
for c in train.columns:
    mean = train[c].mean()
    stdev = train[c].std()

    train[c] = (train[c] - mean) / stdev
    val[c] = (val[c] - mean) / stdev

    mn[c] = mean
    std[c] = stdev

with open(f"models/{FILE_NAME}.pkl", "wb") as file:
    pickle.dump((mn, std), file)

# create sequence
train = SequenceDataset(train, seq_len=SEQ_LEN, shift=SHIFT)
val = SequenceDataset(val, seq_len=SEQ_LEN, shift=SHIFT)

# create batches
train = DataLoader(
    train,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val = DataLoader(val, batch_size=1, shuffle=False)

input_size = len(data.columns)
#########################

if __name__ == "__main__":
    start_time = time.time()
    # Initialize models, loss function, and optimizer
    models = nn.ModuleList(
        [
            Base(LSTMtorch, input_size, hidden_size=300, num_layers=2).to(device),
            # Base(GRUtorch, input_size, hidden_size=300, num_layers=2).to(device),
            # Base(xLSTM, input_size, hidden_size=300, num_layers=2, signature=(1, 1)).to(
            #    device
            # ),
            Base(xLSTM, input_size, hidden_size=300, num_layers=2, signature=(1, 0)).to(
                device
            ),
            Base(xLSTM, input_size, hidden_size=300, num_layers=2, signature=(0, 1)).to(
                device
            ),
            # Base(xGRU, input_size, hidden_size=300, num_layers=2, signature=(1, 1)).to(
            #    device
            # ),
            Base(xGRU, input_size, hidden_size=300, num_layers=2, signature=(1, 0)).to(
                device
            ),
            Base(xGRU, input_size, hidden_size=300, num_layers=2, signature=(0, 1)).to(
                device
            ),
        ]
    )

    # Compile models to increase the performance through epochs
    models = [torch.compile(model) for model in models]

    optimizers = [
        torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) for model in models
    ]
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    for epoch in range(EPOCHS):
        # Train
        train_losses.append(
            [
                train_model(train, model, criterion, optimizer)
                for model, optimizer in zip(models, optimizers)
            ]
        )
        # Val
        val_losses.append([test_model(val, model, criterion) for model in models])

        print(
            f"Epoch [{epoch + 1:03d}/{EPOCHS}] ♪ "
            + " ♦ ".join(
                f"{model.__type__} | Train : {train*100:.2f}% Val : {val*100:.2f}%"
                for model, train, val in zip(models, train_losses[-1], val_losses[-1])
            )
            + f" | Total Time: {(time.time() - start_time) / 60:.2f}m"
        )

        register(np.array(val_losses), models)

    predict_plot(models, val, data.columns, (mn, std))
