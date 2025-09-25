import torch
import torch.nn as nn


class LSTMtorch(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMtorch, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

    @property
    def device(self) -> str:
        return next(self.parameters()).device

    def forward(self, x):
        h = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, device=self.device
        )
        c = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, device=self.device
        )
        return self.lstm(x, (h.detach(), c.detach()))
