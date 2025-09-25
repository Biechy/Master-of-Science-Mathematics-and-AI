import torch
import torch.nn as nn
import torch.functional as F


class Base(nn.Module):
    def __init__(
        self, model, input_size, hidden_size, num_layers, dropout=0.2, **kwargs
    ):
        super(Base, self).__init__()

        self.rnn = model(input_size, hidden_size, num_layers, dropout=dropout, **kwargs)
        self.proj = nn.Linear(hidden_size, input_size)

        # Give a name depending on the layers for xLSTM
        sign = kwargs.pop("signature", None)
        if sign is not None:
            self.__type__ = self.rnn.__class__.__name__ + f"[{sign[0]}:{sign[1]}]"
        else:
            self.__type__ = self.rnn.__class__.__name__

    def forward(self, X):
        out, *res = self.rnn(X)
        out = self.proj(out[:, -1, :])
        return out

    def fit(self, X, y, criterion, optimizer):
        self.train()
        out = self(X)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self(X)
