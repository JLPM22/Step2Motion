import torch
from typing import Callable, Union
from torch import nn
from torch import Tensor
from torch.nn.modules import Linear, Dropout, LayerNorm
from torch.nn import functional as F
from torch.nn.modules.transformer import _get_activation_fn
from insole_multiheadattention import InsoleMultiheadAttention


# This is a copy from PyTorch's TransformerEncoderLayer slightly modified
class TranslationTransformerEncoderLayer(nn.Module):

    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        n_heads = 8
        self.insole_self_attn = InsoleMultiheadAttention(
            d_model, d_model // n_heads, d_model // n_heads, n_heads
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)  # type: ignore
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)  # type: ignore
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
        self,
        x: Tensor,
        memory: list[Tensor],
    ) -> Tensor:

        x = x + self._sa_block(self.norm1(x), memory)
        x = x + self._ff_block(self.norm2(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        mem: list[Tensor],
    ) -> Tensor:
        c = torch.stack(mem, dim=1)
        x = self.insole_self_attn(
            x,
            c,
            c,
        )
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
