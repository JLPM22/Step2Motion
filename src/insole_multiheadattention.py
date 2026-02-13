import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InsoleMultiheadAttention(nn.Module):
    def __init__(self, dmodel: int, dk: int, dv: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dmodel = dmodel
        self.dk = dk

        self.proj_q, self.bias_q = self._get_proj_bias(dk)
        self.proj_k, self.bias_k = self._get_proj_bias(dk)
        self.proj_v, self.bias_v = self._get_proj_bias(dv)

        self.output_proj = nn.Linear(dv * num_heads, dmodel, bias=False)

        self.register_buffer("scale", torch.tensor(dk, dtype=float).sqrt())  # type: ignore

    def _get_proj_bias(self, hidsize):
        proj = nn.Parameter(torch.Tensor(self.num_heads, self.dmodel, hidsize))
        bias = nn.Parameter(torch.Tensor(1, self.num_heads, 1, hidsize))
        nn.init.xavier_uniform_(proj)
        nn.init.constant_(bias, 0.0)
        return proj, bias

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        q: shape [batch, seqlen, d_model]
        k: shape [batch, head, seqlen, d_model]
        v: shape [batch, head, seqlen, d_model]
        """

        q = (q.unsqueeze(1) @ self.proj_q) + self.bias_q
        k = torch.einsum("bhqd,hdk->bhqk", k, self.proj_k) + self.bias_k
        v = torch.einsum("bhqd,hdk->bhqk", v, self.proj_v) + self.bias_v
        # batch, head, seqlen, dk|dv

        heads = F.scaled_dot_product_attention(q, k, v)
        # batch, head, qlen, dv
        hid = torch.cat(heads.unbind(1), -1)
        # batch, qlen, dv * head
        output = self.output_proj(hid)
        # batch, qlen, dmodel
        return output

    def forward_test(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q: shape [batch, seqlen, d_model]
        k: shape [batch, head, seqlen, d_model]
        v: shape [batch, head, seqlen, d_model]
        """

        q = (q.unsqueeze(1) @ self.proj_q) + self.bias_q
        k = torch.einsum("bhqd,hdk->bhqk", k, self.proj_k) + self.bias_k
        v = torch.einsum("bhqd,hdk->bhqk", v, self.proj_v) + self.bias_v
        # batch, head, seqlen, dk|dv

        heads = F.scaled_dot_product_attention(q, k, v)
        # batch, head, qlen, dv
        hid = torch.cat(heads.unbind(1), -1)
        # batch, qlen, dv * head
        output = self.output_proj(hid)
        # batch, qlen, dmodel

        # compute QK^T
        qk = q @ k.transpose(-2, -1)
        # divide by sqrt(dk)
        qk = qk / math.sqrt(self.dk)
        # apply softmax
        attention_matrix = F.softmax(qk, dim=-1)

        return output, attention_matrix
