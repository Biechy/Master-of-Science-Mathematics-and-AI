import torch
import torch.nn as nn


class GRUtorch(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRUtorch, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
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
        return self.gru(x, h.detach())


class GRUcell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUcell, self).__init__()
        self.w = nn.ModuleList(
            [
                nn.Linear(input_size + hidden_size, hidden_size, bias=True)
                for _ in range(3)
            ]
        )

    def forward(self, x, h):
        h = h
        f, i = [w(torch.concat((x, h), dim=-1)) for w in self.w[:2]]

        f = torch.sigmoid(f)
        i = torch.sigmoid(i)

        z = self.w[-1](torch.concat((x, h * i), dim=-1))
        z = torch.tanh(z)

        return (1 - f) * h + f * z


class GRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.first_layer = GRUcell(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList(
            [GRUcell(hidden_size, hidden_size) for _ in range(1, num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        h = torch.zeros(batch_size, self.hidden_size)
        for t in range(seq_len):
            h = self.first_layer(x[:, t, :], h.detach())
            for layer in self.hidden_layers:
                h = self.dropout(h)
                h = layer(h.detach(), h.detach())
        return h
