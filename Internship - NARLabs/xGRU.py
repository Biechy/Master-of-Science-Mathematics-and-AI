import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import einsum, rearrange
from math import sqrt, exp
from torch import Tensor
from typing import Tuple, List
from itertools import repeat

from utils import BlockLinear, CausalConv1d, enlarge_as


def enlarge_as(src: Tensor, other: Tensor) -> Tensor:
    """
    Add sufficient number of singleton dimensions
    to tensor a **to the right** so to match the
    shape of tensor b. NOTE that simple broadcasting
    works in the opposite direction.
    """
    return rearrange(src, f'... -> ...{" 1" * (other.dim() - src.dim())}').contiguous()


class sGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        head_size: int = 16,
        head_num: int = 4,
        ker_size: int = 4,
        p_factor: float = 4 / 3,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.head_size = head_size
        self.head_num = head_num

        self.LN = nn.LayerNorm(input_size)
        self.GN = nn.GroupNorm(head_num, head_size * head_num)

        self.Conv4 = CausalConv1d(1, 1, kernel_size=ker_size)

        self.W = nn.ModuleList([nn.Linear(input_size, head_num * head_size)] * 3)

        self.R = nn.ModuleList([BlockLinear([(head_size, head_size)] * head_num)] * 3)

        # NOTE: The factor of two in the output dimension of the up_proj
        # is due to the fact that the output needs to branch into two
        # separate outputs to account for the the gated GeLU connection.
        # See Fig. 9 in the paper.
        proj_dim = int(p_factor * head_num * head_size)
        self.up_PF_left, self.up_PF_right = [
            nn.Linear(head_num * head_size, proj_dim)
        ] * 2
        self.down_PF = nn.Linear(proj_dim, input_size)

    @property
    def device(self) -> str:
        """Get the device of the model.

        Returns:
            str: The device of the model.
        """
        return next(self.parameters()).device

    def init_hidden(self, bs: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Initialize the hidden state of the sGRU model.

        Args:
            batch_size (int): The batch size of the input sequence.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: The hidden state tuple containing the cell state,
                normalizer state, hidden state, and stabilizer state.
        """

        n = torch.ones(bs, self.head_num * self.head_size, device=self.device)
        h, m = [torch.zeros(bs, self.head_num * self.head_size, device=self.device)] * 2
        return n, h, m

    def forward(
        self,
        seq: Tensor,
        hid: Tuple[Tensor, Tensor, Tensor],
        use_conv: bool = False,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """Forward pass of the sGRU model.

        Args:
            seq (Tensor): The input sequence tensor of shape (batch_size, input_dim).
            hid (Tuple[Tensor, Tensor, Tensor]): The hidden state tuple containing the cell state,
                normalizer state, hidden state, and stabilizer state.

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]: The output tensor with the residual
                connection and the newly updated hidden state tuple.
        """

        n_, h_, m_ = hid

        x: Tensor = self.LN(seq)

        if use_conv:
            # ! Not working
            x_conv = self.Conv4(x)
            x_conv = F.silu(x_conv).squeeze()
        else:
            x_conv = x

        # NOTE: For input (i) and forget (f) inputs we use
        # the output of the causal conv. See Fig. 9 in the paper.
        i, f = [W(x_conv) + R(h_) for W, R in zip(self.W[:2], self.R[:2])]

        # Stabilize gates with an additional state m
        m = torch.max(f + m_, i)  # Eq. (15) in ref. paper
        i = torch.exp(i - m)  # Eq. (16) in ref. paper | or Eq. (38) in supp. mat.
        f = torch.exp(f - m + m_)  # Eq. (17) in ref. paper | or Eq. (39) in supp. mat.

        z = self.W[2](x) + self.R[2](i * h_)  # ? GRU changement

        z = torch.tanh(z)  # Eq. (11) in ref. paper

        # Update the internal states of the model
        n = f * n_ + i  # Eq. (9) in ref. paper
        h = (exp(1) - f) * h_ + (f * z)  # ? GRU changement

        out = self.GN(h)

        # Projection factor 4/3. See Fig. (9) in supp. mat.
        left, right = self.up_PF_right(out), self.up_PF_left(out)
        out = left + F.gelu(right)
        out = self.down_PF(out)

        return out + seq, (n, h, m)


class mGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        head_size: int = 16,
        head_num: int = 4,
        p_factor: int = 2,
        ker_size: int = 4,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.head_num = head_num
        self.head_size = head_size

        hid_dim = head_num * head_size

        self.LN = nn.LayerNorm(input_size)
        self.GN = nn.GroupNorm(head_num, hid_dim)

        # NOTE: The factor of two in the output dimension of the up_proj
        # is due to the fact that the output needs to branch into two
        self.up_PF_left = nn.Linear(input_size, int(p_factor * input_size))
        self.up_PF_right = nn.Linear(input_size, hid_dim)
        self.down_PF = nn.Linear(hid_dim, input_size)

        self.Conv4 = CausalConv1d(1, 1, kernel_size=ker_size)

        self.LSkip = nn.Conv1d(
            int(p_factor * input_size), hid_dim, kernel_size=1, bias=False
        )

        self.W_i = nn.Linear(int(p_factor * input_size), head_num)
        self.W_o = nn.Linear(int(p_factor * input_size), hid_dim)

        self.W_q = nn.Linear(int(p_factor * input_size), hid_dim)
        self.W_k = nn.Linear(int(p_factor * input_size), hid_dim)
        self.W_v = nn.Linear(int(p_factor * input_size), hid_dim)

    @property
    def device(self) -> str:
        """Get the device of the model.

        Returns:
            str: The device of the model.
        """
        return next(self.parameters()).device

    def init_hidden(self, bs: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Initialize the hidden state of the sGRU model.

        Args:
            batch_size (int): The batch size of the input sequence.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: The hidden state tuple containing the cell state,
                normalizer state, hidden state, and stabilizer state.
        """

        c = torch.zeros(
            bs, self.head_num, self.head_size, self.head_size, device=self.device
        )
        n = torch.ones(bs, self.head_num, self.head_size, device=self.device)
        m = torch.zeros(bs, self.head_num, device=self.device)

        return c, n, m

    def forward(
        self,
        seq: Tensor,
        hid: Tuple[Tensor, Tensor, Tensor],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """_summary_

        Args:
            seq (Tensor): _description_
            hid (Tuple[Tensor, Tensor, Tensor]): _description_

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]: _description_
        """
        # Separate the hidden (previous) state into the cell state,
        # the normalizer state, the hidden state, and the stabilizer state.
        c_, n_, m_ = hid
        x: Tensor = self.LN(seq)  # shape: b i
        left = self.up_PF_left(x)  # shape: b (i * p_factor)
        right = self.up_PF_right(x)  # shape: b (h d)

        # Compute the causal convolutional input (to be
        # used for the query and key gates)
        x_conv = self.Conv4(left)  # shape: b 1 (i * p_factor)
        x_conv = F.silu(x_conv).squeeze(1)  # shape: b (i * p_factor)
        # ! Pourquoi c'est un simple linear et pas une blockdiag ?
        q = rearrange(self.W_q(x_conv), "b (h d) -> b h d", h=self.head_num)
        k = rearrange(self.W_k(x_conv), "b (h d) -> b h d", h=self.head_num) / sqrt(
            self.head_size
        )
        v = rearrange(self.W_v(left), "b (h d) -> b h d", h=self.head_num)

        i: Tensor = self.W_i(x_conv)  # shape: b h
        f: Tensor = 1 - i  # ? GRU changement
        o: Tensor = self.W_o(left)  # shape: b (h d)

        # Stabilize gates with an additional state m
        m = torch.max(f + m_, i)
        i = torch.exp(i - m)  # Eq. (25) in ref. paper
        f = torch.exp(f - m + m_)  # Eq. (26) in ref. paper
        o = torch.sigmoid(o)  # Eq. (27) in ref. paper

        # Update the internal states of the model
        c = enlarge_as(f, c_) * c_ + enlarge_as(i, c_) * einsum(
            v, k, "b h d, b h p -> b h d p"
        )
        n = enlarge_as(f, n_) * n_ + enlarge_as(i, k) * k
        h = o * rearrange(
            einsum(c, q, "b h d p, b h p -> b h d")
            / einsum(n, q, "b h d, b h d -> b h").clamp(min=1).unsqueeze(-1),
            "b h d -> b (h d)",
        )  # Eq. (21) in ref. paper

        x_conv = rearrange(x_conv, "b i -> b i 1")
        out = self.GN(h) + self.LSkip(x_conv).squeeze()  # shape: b (h d)
        out = out * F.silu(right)  # shape: b (h d)
        out = self.down_PF(out)  # shape: h i

        return out + seq, (c, n, m)


class xGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        signature: Tuple[int, int],
        dropout: float,
        head_size: int = 16,
        head_num: int = 4,
        p_factor: Tuple[float, float] = (2, 4 / 3),
        ker_size: int = 4,
    ) -> None:
        """Initialize the LLM model.

        Args:
            vocab_size (int): The size of the vocabulary.
            num_layers (int): The number of layers in the LLM model.
            signature (Tuple[int, int]): The signature of the LLM model,
                which represents the ration of the mGRU-to-sGRU blocks.
            hidden_size (int): The dimension of each attention head.
            input_size (int): The dimension of the input tokens.
            head_num (int): The number of attention heads.
            p_factor (Tuple[float, float], optional): The expansion factor
                for the MLP projection in the m|s-LSTM blocks. Defaults to (2, 4/3).
            ker_size (int, optional): The kernel size for the causal convolutional layers.
                Defaults to 4.

            kwargs: Additional keyword arguments used at inference time (see relevant
                arguments of the generate method).
        """
        super().__init__()

        m_factor, s_factor = p_factor

        mgru_config = {
            "input_size": hidden_size,
            "head_size": head_size,
            "head_num": head_num,
            "p_factor": m_factor,
            "ker_size": ker_size,
        }

        sgru_config = {
            "input_size": hidden_size,
            "head_size": head_size,
            "head_num": head_num,
            "p_factor": s_factor,
            "ker_size": ker_size,
        }

        m_num, s_num = signature
        which = [True] * m_num + [False] * s_num

        self.layers: List[mGRU | sGRU] = nn.ModuleList(
            [
                mGRU(**mgru_config) if v else sGRU(**sgru_config)
                for w in repeat(which, num_layers)
                for v in w
            ]
        )
        self.proj = nn.Linear(input_size, hidden_size, bias=False)

    def forward(
        self,
        seq: Tensor,
        hid: List[Tuple[Tensor, ...]] | None = None,
        batch_first: bool = True,
    ) -> Tuple[Tensor, List[Tuple[Tensor, ...]]]:
        """Forward pass of the xLSTM model.

        Args:
            seq (Tensor): Input tensor representing the sequence seqens.
                Expected shape: (batch, seq_len) if batch_first=True,
                else (seq_len, batch).
            hid (Hidden, optional): Cache object for storing intermediate hidden
                values of the m|s-LSTM blocks of the model. If None, the hidden
                states are initialized by the models. Defaults to None.

        Returns:
            Tuple[Tensor, Hidden]: Returns tensor of predicted logits of shape
                (batch, seq_len, vocab_size) if batch_first=True or of shape
                (seq_len, batch, vocab_size) if batch_first=False, and the
                updated hidden model states.
        """

        seq: Tensor = torch.atleast_2d(seq)
        seq: Tensor = self.proj(seq)
        if batch_first:
            seq = rearrange(seq, "b s i -> s b i")
        if hid is None:
            hid = [l.init_hidden(seq.shape[1]) for l in self.layers]

        # Pass the sequence through the mGRU and sGRU blocks
        out = []
        for inp in seq:
            # Compute model output and update the hidden states
            for i, lstm in enumerate(self.layers):
                inp, hid[i] = lstm(inp, tuple(h.detach() for h in hid[i]))

            out.append(inp)

        out = torch.stack(out, dim=1 if batch_first else 0)  #  (S, B, I) -> (B, S, I)

        return out, hid
