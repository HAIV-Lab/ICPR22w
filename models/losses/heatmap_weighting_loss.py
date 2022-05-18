import torch
import torch.nn as nn

from ..builder import LOSSES


@LOSSES.register_module()
class HeatmapWeightingMSELoss(nn.Module):

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        

    def forward(self, output, target, target_weight):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            """
            Set different weight generation functions.
            """
            weight = heatmap_gt + 1
            """
            weight = heatmap_gt * 2 + 1
            weight = heatmap_gt * heatmap_gt + 1
            """
            
            if self.use_target_weight:
                loss += torch.mean(self.criterion(heatmap_pred * target_weight[:, idx],
                                       heatmap_gt * target_weight[:, idx]) * weight)
            else:
                loss += torch.mean(self.criterion(heatmap_pred, heatmap_gt) * weight)

        return loss / num_joints * self.loss_weight
