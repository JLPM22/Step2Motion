import torch
import torch.nn as nn
from typing import Tuple


class ForwardDiffusion(nn.Module):
    def __init__(self, beta_start: float = 1e-4, beta_end: float = 0.02, T: int = 200) -> None:
        super(ForwardDiffusion, self).__init__()

        self.T = T

        # Pre-calculate the different term for the closed equation:
        # q(x_t, x_0) = N(x_t; sqrt(cum_alpha_t) * x_0, (1 - cum_alpha_t) * I)
        # x_t = sqrt(cum_alpha_t) * x_0 + sqrt(1 - cum_alpha_t) * N(0, I)
        betas = self.beta_scheduler(beta_start, beta_end)
        betas = torch.cat([torch.tensor([0.0]), betas], dim=0).unsqueeze(-1).unsqueeze(-1)
        alphas = 1.0 - betas
        cum_alphas = torch.cumprod(alphas, dim=0)
        sqrt_cum_alphas = torch.sqrt(cum_alphas)
        sqrt_one_minus_cum_alphas = torch.sqrt(1.0 - cum_alphas)
        sqrt_alphas = torch.sqrt(alphas)
        recriprocal_sqrt_alphas = 1.0 / sqrt_alphas
        one_minus_alpha_div_sqrt_one_minus_cum_alphas = (1.0 - alphas) / sqrt_one_minus_cum_alphas
        # variance posterior q(x_{t-1} | x_t, x_0)
        cum_alphas_prev = torch.cat([torch.tensor([[[float("nan")]]]), cum_alphas[:-1]], dim=0)
        posterior_variance = betas * (1.0 - cum_alphas_prev) / (1.0 - cum_alphas)
        posterior_std = torch.sqrt(posterior_variance)
        # coefficient 1 for posterior q(x_{t-1} | x_t, x_0)
        posterior_coeff_1 = betas * torch.sqrt(cum_alphas_prev) / (1.0 - cum_alphas)
        # coefficient 2 for posterior q(x_{t-1} | x_t, x_0)
        posterior_coeff_2 = sqrt_alphas * (1.0 - cum_alphas_prev) / (1.0 - cum_alphas)

        # shape: (T + 1, 1) x_0, x_1 ... x_T
        self.register_buffer("sqrt_cum_alphas", sqrt_cum_alphas)
        self.register_buffer("sqrt_one_minus_cum_alphas", sqrt_one_minus_cum_alphas)
        self.register_buffer("recriprocal_sqrt_alphas", recriprocal_sqrt_alphas)
        self.register_buffer(
            "one_minus_alpha_div_sqrt_one_minus_cum_alphas", one_minus_alpha_div_sqrt_one_minus_cum_alphas
        )
        self.register_buffer("posterior_std", posterior_std)
        self.register_buffer("posterior_coeff_1", posterior_coeff_1)
        self.register_buffer("posterior_coeff_2", posterior_coeff_2)

    def beta_scheduler(self, beta_start: float, beta_end: float) -> torch.Tensor:
        def linear_schedule() -> torch.Tensor:
            return torch.linspace(beta_start, beta_end, self.T)

        # TODO: Research and implement other schedules
        return linear_schedule()

    def forward(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process
        :param x_0: Initial state, shape (batch_size, T, n_features)
        :param t: Time step, shape (batch_size,)
        """
        device = x_0.device
        noise = torch.randn_like(x_0, device=device)
        x_t = self.sqrt_cum_alphas[t] * x_0 + self.sqrt_one_minus_cum_alphas[t] * noise
        return x_t, noise

    def q_posterior_mean_from_noise(
        self, x_t: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample x_t_minus_one from the posterior mean given x_t and noise
        x_t_minus_one ~ q(x_t_minus_one | x_t) ~ p_theta(x_t_minus_one | x_t)
        :param x_t: Current state, shape (batch_size, T, n_features)
        :param noise: Noise, shape (batch_size, T, n_features)
        :param t: Time step, shape (batch_size,)
        """
        # assert all(t > 0), "t must be greater than 0"
        z = torch.randn_like(x_t, device=x_t.device)
        mean_theta = self.recriprocal_sqrt_alphas[t] * (
            x_t - self.one_minus_alpha_div_sqrt_one_minus_cum_alphas[t] * noise
        )
        return mean_theta + self.posterior_std[t] * z

    def q_posterior_mean_from_x_0(
        self, x_t: torch.Tensor, x_0: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample x_t_minus_one from the posterior mean given x_t and x_0
        x_t_minus_one ~ q(x_t_minus_one | x_t, x_0)
        :param x_t: Current state, shape (batch_size, T, n_features)
        :param x_0: Initial state, shape (batch_size, T, n_features)
        :param t: Time step, shape (batch_size,)
        """
        # assert all(t > 0), "t must be greater than 0"
        z = torch.randn_like(x_t, device=x_t.device)
        mean_theta = self.posterior_coeff_1[t] * x_0 + self.posterior_coeff_2[t] * x_t
        return mean_theta + self.posterior_std[t] * z
