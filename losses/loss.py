# losses/loss.py
import torch
import torch.nn as nn


class AtmosLoss(torch.nn.Module):

    def __init__(self, weights=None, vturb_zero_weight=0.0):
        super().__init__()

        self.weights = weights or {
            "temp": 1.0,
            "vlos": 1.0,
            "vturb": 1.0,
            "blong": 1.0,
        }

        self.vturb_zero_weight = vturb_zero_weight
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, target):

        lt = pred.shape[1] // 4

        p_temp  = pred[:, :lt]
        p_vlos  = pred[:, lt:2*lt]
        p_vturb = pred[:, 2*lt:3*lt]
        p_blong = pred[:, 3*lt:]

        t_temp  = target[:, :lt]
        t_vlos  = target[:, lt:2*lt]
        t_vturb = target[:, 2*lt:3*lt]
        t_blong = target[:, 3*lt:]

        loss_temp  = self.mse(p_temp,  t_temp)
        loss_vlos  = self.mse(p_vlos,  t_vlos)
        loss_vturb = self.mse(p_vturb, t_vturb)
        loss_blong = self.mse(p_blong, t_blong)

        # -------------------------
        # ZERO-FORCING vturb penalty
        # -------------------------
        if self.vturb_zero_weight > 0:
            zero_penalty = torch.mean(p_vturb**2)
        else:
            zero_penalty = 0.0

        total = (
            self.weights["temp"]  * loss_temp +
            self.weights["vlos"]  * loss_vlos +
            self.weights["blong"] * loss_blong +
            self.weights["vturb"] * loss_vturb +
            self.vturb_zero_weight * zero_penalty
        )

        return total
