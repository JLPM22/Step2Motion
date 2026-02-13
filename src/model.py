import torch
import torch.nn as nn
from typing import Optional
from prior_transformer_decoder_layer import PriorTransformerDecoderLayer
from control_transformer_decoder_layer import ControlTransformerDecoderLayer
from utils import SinusoidalPositionalEncoding


class TransformerTranslation(nn.Module):

    def __init__(
        self,
        c_dim: int,
        input_T: int,
        output_dim: int,
        hidden_dim: int,
        n_hidden: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        is_prior: bool,
    ) -> None:
        super(TransformerTranslation, self).__init__()

        self.set_normalizer_id(0)
        self.translation_dim = 3
        self.pose_dim = output_dim - self.translation_dim
        self.is_prior = is_prior

        self.temporal_pe = SinusoidalPositionalEncoding(max_length=input_T, d_model=hidden_dim)

        self.input_emb = nn.Sequential(
            nn.Linear(12, hidden_dim),
            # nn.Linear(50, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_dim, n_heads, d_ff, dropout, batch_first=True, activation="gelu"
            ),
            n_hidden,
        )

        self.linear = nn.Linear(hidden_dim, self.translation_dim)

    def forward(self, pose: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        :param pose: input tensor of shape (batch_size, T, pose_dim)
        :param c: condition tensor of shape (batch_size, T, c_dim)
        :return: output tensor of shape (batch_size, T, translation_dim)
        """
        # x = torch.cat([pose, c], dim=-1)
        x = c[..., [16, 17, 18, 19, 20, 21, 41, 42, 43, 44, 45, 46]]
        # x = c
        x = self.input_emb(x)
        x = x + self.temporal_pe.forward_temporality(x.shape[1])
        translation = self.encoder(x)
        translation = self.linear(translation)
        return translation

    def set_normalizer_id(self, id: int) -> None:
        self.register_buffer("normalizer_id", torch.tensor(id))


class ControlTransformer(nn.Module):

    def __init__(
        self,
        input_T: int,
        c_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_hidden: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super(ControlTransformer, self).__init__()

        self.set_normalizer_id(0)
        self.translation_dim = 3
        self.pose_dim = output_dim - self.translation_dim
        self.distances_dim = self.pose_dim // 3
        self.c_dim = c_dim
        d_model = hidden_dim

        self.temporal_pe = SinusoidalPositionalEncoding(max_length=input_T, d_model=d_model)

        self.condition_emb_lpressure_toes = nn.Sequential(
            nn.Linear(8, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.condition_emb_lpressure_heel = nn.Sequential(
            nn.Linear(8, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.condition_emb_limu = nn.Sequential(
            nn.Linear(6, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.condition_emb_lothers = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.condition_emb_rpressure_toes = nn.Sequential(
            nn.Linear(8, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.condition_emb_rpressure_heel = nn.Sequential(
            nn.Linear(8, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.condition_emb_rimu = nn.Sequential(
            nn.Linear(6, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.condition_emb_rothers = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # self.distances_emb = nn.Linear(self.distances_dim, hidden_dim)
        # self.encoder = ControlTransformerEncoder(
        #     ControlTransformerEncoderLayer(
        #         hidden_dim, n_heads, d_ff, dropout, batch_first=True, activation="gelu"
        #     ),
        #     n_hidden,
        # )
        self.decoder_layers = nn.ModuleList(
            [
                ControlTransformerDecoderLayer(d_model, n_heads, dropout, batch_first=True, activation="gelu")
                for _ in range(n_hidden)
            ]
        )

    def forward_condition_emb(
        self, c: torch.Tensor, distances: torch.Tensor, t_emb: torch.Tensor, c_mask: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Forward pass of the model.
        :param c: condition tensor of shape (batch_size, T, c_dim)
        :param distances: distances tensor of shape (batch_size, distances_dim)
        :param t_emb: diffusion timestep tensor of shape (batch_size, 1, hidden_dim)
        :param c_mask: mask tensor of shape (batch_size,)
        :return: output tensor of shape (batch_size, T, hidden_dim)
        """
        temporality = c.shape[1]

        c_lpressure_toes = self.condition_emb_lpressure_toes(c[..., 8:16]) * c_mask.unsqueeze(-1).unsqueeze(
            -1
        )
        c_lpressure_heel = self.condition_emb_lpressure_heel(c[..., 0:8]) * c_mask.unsqueeze(-1).unsqueeze(-1)
        c_limu = self.condition_emb_limu(c[..., 16:22]) * c_mask.unsqueeze(-1).unsqueeze(-1)
        c_lothers = self.condition_emb_lothers(c[..., 22:25]) * c_mask.unsqueeze(-1).unsqueeze(-1)
        c_rpressure_toes = self.condition_emb_rpressure_toes(c[..., 33:41]) * c_mask.unsqueeze(-1).unsqueeze(
            -1
        )
        c_rpressure_heel = self.condition_emb_rpressure_heel(c[..., 25:33]) * c_mask.unsqueeze(-1).unsqueeze(
            -1
        )
        c_rimu = self.condition_emb_rimu(c[..., 41:47]) * c_mask.unsqueeze(-1).unsqueeze(-1)
        c_rothers = self.condition_emb_rothers(c[..., 47:50]) * c_mask.unsqueeze(-1).unsqueeze(-1)

        c_lpressure_toes = c_lpressure_toes + self.temporal_pe.forward_temporality(temporality)
        c_lpressure_heel = c_lpressure_heel + self.temporal_pe.forward_temporality(temporality)
        c_limu = c_limu + self.temporal_pe.forward_temporality(temporality)
        c_lothers = c_lothers + self.temporal_pe.forward_temporality(temporality)
        c_rpressure_toes = c_rpressure_toes + self.temporal_pe.forward_temporality(temporality)
        c_rpressure_heel = c_rpressure_heel + self.temporal_pe.forward_temporality(temporality)
        c_rimu = c_rimu + self.temporal_pe.forward_temporality(temporality)
        c_rothers = c_rothers + self.temporal_pe.forward_temporality(temporality)
        return [
            c_lpressure_toes,
            c_lpressure_heel,
            c_limu,
            c_lothers,
            c_rpressure_toes,
            c_rpressure_heel,
            c_rimu,
            c_rothers,
        ]

    def forward_decoder_layer(
        self, x: torch.Tensor, c_emb: list[torch.Tensor], layer: int, test: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.
        :param x: input tensor of shape (batch_size, T, hidden_dim)
        :param c_emb: condition tensor of shape (batch_size, T, hidden_dim)
        :param layer: decoder layer to execute
        :return: output tensor of shape (batch_size, T, hidden_dim)
        """
        return self.decoder_layers[layer](x, c_emb, test=test)

    def set_normalizer_id(self, id: int) -> None:
        self.register_buffer("normalizer_id", torch.tensor(id))


class PriorTransformer(nn.Module):

    def __init__(
        self,
        input_T: int,
        output_dim: int,
        hidden_dim: int,
        n_hidden: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        diffusion_T: int,
    ) -> None:
        super(PriorTransformer, self).__init__()

        self.set_normalizer_id(0)
        self.translation_dim = 3
        self.pose_dim = output_dim - self.translation_dim
        self.n_joints = self.pose_dim // 3
        # self.distances_dim = self.pose_dim // 3

        self.left_leg_emb = nn.Linear(3 * 4, hidden_dim)
        self.right_leg_emb = nn.Linear(3 * 4, hidden_dim)
        self.body_emb = nn.Linear(3 * (self.n_joints - 8), hidden_dim)
        # self.distances_emb = nn.Linear(self.distances_dim, hidden_dim)
        self.timestep_emb = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temporal_pe = SinusoidalPositionalEncoding(
            max_length=max(input_T, diffusion_T + 1), d_model=hidden_dim
        )
        self.decoder_layers = nn.ModuleList(
            [
                PriorTransformerDecoderLayer(
                    hidden_dim, n_heads, d_ff, dropout, batch_first=True, activation="gelu"
                )
                for _ in range(n_hidden)
            ]
        )
        self.out_proj_left_leg = nn.Linear(hidden_dim, 3 * 4)
        self.out_proj_right_leg = nn.Linear(hidden_dim, 3 * 4)
        self.out_proj_body = nn.Linear(hidden_dim, 3 * (self.n_joints - 8))

    def forward(
        self,
        x: torch.Tensor,
        distances: torch.Tensor,
        t: torch.Tensor,
        control: Optional[ControlTransformer],
        c: Optional[torch.Tensor],
        c_mask: Optional[torch.Tensor],
        test: bool = False,
    ) -> tuple[torch.Tensor, list[Optional[torch.Tensor]]]:
        """
        Forward pass of the model.
        :param x: input tensor of shape (batch_size, T, pose_dim)
        :param distances: distances tensor of shape (batch_size, distances_dim)
        :param t: diffusion timestep tensor of shape (batch_size,)
        :param control: control transformer model
        :param c: condition tensor of shape (batch_size, T, c_dim)
        :param c_mask: mask tensor of shape (batch_size,)
        :return: output tensor of shape (batch_size, T, pose_dim)
        """
        # Prepare diffusion timestep and conditioning
        t = self.temporal_pe.forward_timestep(t)
        t_emb = self.timestep_emb(t).unsqueeze(1)
        # Conditioning
        if control is not None and c is not None and c_mask is not None:
            c_emb = control.forward_condition_emb(c, distances, t_emb, c_mask)
        # Pose prediction
        temporality = x.shape[1]
        left_leg = x[..., : 4 * 3]
        right_leg = x[..., 4 * 3 : 8 * 3]
        body = x[..., 8 * 3 :]
        left_leg = self.left_leg_emb(left_leg)
        right_leg = self.right_leg_emb(right_leg)
        body = self.body_emb(body)
        # Distances
        # distances = self.distances_emb(distances)
        # distances = distances.unsqueeze(1)
        # TODO: some people apply dropout here after the summation, test?
        left_leg = left_leg + self.temporal_pe.forward_temporality(temporality)
        right_leg = right_leg + self.temporal_pe.forward_temporality(temporality)
        body = body + self.temporal_pe.forward_temporality(temporality)
        pose = torch.cat([left_leg, right_leg, body], dim=1)
        attn_mats = []
        for i, layer in enumerate(self.decoder_layers):
            control_layer = None
            if control is not None and c_emb is not None:
                control_layer = lambda x: control.forward_decoder_layer(x, c_emb, i, test=test)
            pose, attn_mat = layer(pose, distances, control_layer, t_emb)
            attn_mats.append(attn_mat)
        # Project to output dimensions
        left_leg = pose[..., :temporality, :]
        right_leg = pose[..., temporality : 2 * temporality, :]
        body = pose[..., 2 * temporality :, :]
        left_leg = self.out_proj_left_leg(left_leg)
        right_leg = self.out_proj_right_leg(right_leg)
        body = self.out_proj_body(body)
        pose = torch.cat([left_leg, right_leg, body], dim=-1)
        return pose, attn_mats

    def set_normalizer_id(self, id: int) -> None:
        self.register_buffer("normalizer_id", torch.tensor(id))
