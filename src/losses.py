import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseLossPrior(nn.Module):

    def __init__(self, device: torch.device) -> None:
        super(PoseLossPrior, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, insole: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class PoseLoss(nn.Module):

    def __init__(self, device: torch.device) -> None:
        super(PoseLoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, insole: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class TranslationLoss(nn.Module):
    def __init__(self) -> None:
        super(TranslationLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_cumsum = torch.cumsum(pred, dim=-2)
        target_cumsum = torch.cumsum(target, dim=-2)
        return self.loss(pred, target) + self.loss(pred_cumsum, target_cumsum) * 0.001
