import torch
import torch.nn as nn


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='none')

    def forward(self, pred, target, mask=None):
        # pred, target: (B, 2K)
        loss = self.l1(pred, target).mean(dim=1)  # per-sample
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum()+1e-6)
        return loss.mean()