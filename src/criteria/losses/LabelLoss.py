import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelLoss(nn.Module):
    def __init__(
        self,
        loss_weight: int = 1,
    ):
        super().__init__()

        self.loss_weight = loss_weight

    def forward(self, outputs, targets, indices, **kwargs):
        idx = kwargs["src_permutation_idx"]

        src_logits = outputs["pred_logits"][idx]
        target_classes = torch.cat(
            [t["labels"][i] for t, (_, i) in zip(targets, indices)], dim=0
        ).float()

        loss = F.cross_entropy(src_logits, target_classes)

        return loss * self.loss_weight
