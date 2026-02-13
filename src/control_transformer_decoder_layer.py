import torch
import torch.nn.functional as F
from insole_multiheadattention import InsoleMultiheadAttention
from typing import Callable, Optional, Union
from torch import Tensor
from torch import nn
from torch.nn.modules import Dropout, LayerNorm
from torch.nn.modules.transformer import _get_activation_fn


# This is a copy from PyTorch's TransformerDecoderLayer slightly modified
class ControlTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # self.cross_attn = MultiheadAttention(
        #     d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias, **factory_kwargs  # type: ignore
        # )
        n_heads = 8
        self.cross_attn = InsoleMultiheadAttention(d_model, d_model // n_heads, d_model // n_heads, n_heads)

        self.norm = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)  # type: ignore
        self.dropout = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)  # type: ignore

    def forward(
        self,
        x: Tensor,
        memory: list[Tensor],
        test: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        attn_x, attn_mat = self._mha_block(self.norm(x), memory, test=test)
        x = x + attn_x
        return x, attn_mat

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: list[Tensor],
        test: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        c = torch.stack(mem, dim=1)
        attn_mat = None
        if test:
            x, attn_mat = self.cross_attn.forward_test(x, c, c)
        else:
            x = self.cross_attn(x, c, c)
        return self.dropout(x), attn_mat
