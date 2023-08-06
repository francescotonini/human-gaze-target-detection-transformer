import torch

from torchmetrics.detection.mean_ap import MeanAveragePrecision


class BoxesAP:
    def __init__(self):
        super().__init__()

        self.metric = MeanAveragePrecision(
            box_format="cxcywh",
            iou_type="bbox",
            class_metrics=False,
        )

    def reset_metrics(self):
        self.metric.reset()

    def get_metrics(self):
        # This is already a dict. Add "boxes" to the key
        return {"boxes_" + k: v.item() for k, v in self.metric.compute().items()}

    @torch.no_grad()
    def __call__(self, outputs, targets, indices, **kwargs):
        # If metric is not on the same device as outputs, put it
        # on the same device as outputs
        if self.metric.device != outputs["pred_logits"].device:
            self.metric = self.metric.to(outputs["pred_logits"].device)

        idx = kwargs["src_permutation_idx"]

        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        target_labels = torch.cat(
            [t["labels"][i] for t, (_, i) in zip(targets, indices)], dim=0
        ).argmax(-1)

        src_boxes = outputs["pred_boxes"][idx]
        src_labels = outputs["pred_logits"][idx].argmax(-1)
        src_scores = outputs["pred_logits"][idx].softmax(-1).max(-1).values

        preds = [
            {
                "boxes": src_boxes,
                "labels": src_labels,
                "scores": src_scores,
            }
        ]

        targets = [
            {
                "boxes": target_boxes,
                "labels": target_labels,
            }
        ]

        self.metric(preds, targets)
