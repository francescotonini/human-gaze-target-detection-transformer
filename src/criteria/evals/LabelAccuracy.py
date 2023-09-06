import torch
from torchmetrics import Accuracy


class LabelAccuracy:
    def __init__(self):
        super().__init__()

        self.metric = Accuracy(task="binary")

    def reset_metrics(self):
        self.metric.reset()

    def get_metrics(self):
        return {
            "label_accuracy": self.metric.compute().item(),
        }

    @torch.no_grad()
    def __call__(self, outputs, targets, indices, **kwargs):
        # If metric is not on the same device as outputs, put it
        # on the same device as outputs
        if self.metric.device != outputs["pred_logits"].device:
            self.metric = self.metric.to(outputs["pred_logits"].device)

        idx = kwargs["src_permutation_idx"]

        pred_class = outputs["pred_logits"][idx].argmax(-1)
        tgt_class = torch.cat(
            [t["labels"][j] for t, (_, j) in zip(targets, indices)]
        ).argmax(-1)

        self.metric(pred_class, tgt_class)
