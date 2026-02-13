import torch.nn.functional as F
from typing import Callable, Optional, Union
from torch import Tensor
from torch import nn
from torch.nn.modules import MultiheadAttention, Linear, Dropout, LayerNorm
from torch.nn.modules.transformer import _get_activation_fn


# This is a copy from PyTorch's TransformerDecoderLayer slightly modified
class PriorTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
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

        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias, **factory_kwargs  # type: ignore
        )
        # self.skeleton = nn.Sequential(
        #     nn.Linear(d_model, d_model),
        #     nn.GELU(),
        #     nn.Linear(d_model, d_model),
        # )
        # self.skeleton_attn = MultiheadAttention(
        #     d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias, **factory_kwargs  # type: ignore
        # )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        self.linear_tmp = Linear(d_model, d_model, bias=bias, **factory_kwargs)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)  # type: ignore
        # self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)  # type: ignore
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)  # type: ignore
        self.dropout1 = Dropout(dropout)
        # self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

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
        tgt: Tensor,
        distances: Tensor,
        control: Optional[Callable[[Tensor], Tensor]],
        diffusion_T: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:

        x = tgt
        x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
        x = x + self.linear_tmp(diffusion_T)
        attn_mat = None
        if control is not None:
            x, attn_mat = control(x)
        # distances = self.skeleton(distances)
        # x = x + self._mha_block(self.norm2(x), distances, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
        x = x + self._ff_block(self.norm3(x))
        return x, attn_mat

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.skeleton_cross_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
