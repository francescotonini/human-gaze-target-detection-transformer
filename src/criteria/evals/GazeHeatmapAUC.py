import torch
import torchvision.transforms.functional as TF
from torchmetrics import MeanMetric
from torchmetrics.functional import auroc

from src.utils import box_ops
from src.utils.gaze_ops import get_multi_hot_map


class GazeHeatmapAUC:
    def __init__(self, gaze_heatmap_size: int = 64):
        super().__init__()

        self.gaze_heatmap_size = gaze_heatmap_size
        self.metric = MeanMetric()

    def reset_metrics(self):
        self.metric.reset()

    def get_metrics(self):
        return {
            "gaze_heatmap_auc": self.metric.compute().item(),
        }

    @torch.no_grad()
    def __call__(self, outputs, targets, indices, **kwargs):
        # If metric is not on the same device as outputs, put it
        # on the same device as outputs
        if self.metric.device != outputs["pred_logits"].device:
            self.metric = self.metric.to(outputs["pred_logits"].device)

        idx = kwargs["src_permutation_idx"]

        tgt_regression_padding = torch.cat(
            [t["regression_padding"][i] for t, (_, i) in zip(targets, indices)], dim=0
        ).squeeze(1)
        tgt_gaze_points = torch.cat(
            [t["gaze_points"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )[~tgt_regression_padding]
        tgt_gaze_points_padding = torch.cat(
            [t["gaze_points_padding"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )[~tgt_regression_padding]
        tgt_watch_outside = torch.cat(
            [t["gaze_watch_outside"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )[~tgt_regression_padding].bool()
        tgt_bbox = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )[~tgt_regression_padding]

        pred_bbox = outputs["pred_boxes"][idx][~tgt_regression_padding]
        pred_heatmaps = outputs["pred_gaze_heatmap"][idx].reshape(
            -1, self.gaze_heatmap_size, self.gaze_heatmap_size
        )[~tgt_regression_padding]

        img_sizes = torch.cat(
            [t["img_size"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )[~tgt_regression_padding].reshape(-1, 2)

        paths = [
            [t["path"] for _ in range(len(i))] for t, (_, i) in zip(targets, indices)
        ]
        paths = [item for sublist in paths for item in sublist]
        paths = [p for p, t in zip(paths, tgt_regression_padding) if not t]

        for idx, (
            pred_heatmap,
            tgt_gaze_point,
            tgt_gaze_point_padding,
            tgt_watch_outside,
            img_size,
            pred_bbox,
            tgt_bbox,
            path,
        ) in enumerate(
            zip(
                pred_heatmaps,
                tgt_gaze_points,
                tgt_gaze_points_padding,
                tgt_watch_outside,
                img_sizes,
                pred_bbox,
                tgt_bbox,
                paths,
            )
        ):
            if tgt_watch_outside:
                continue

            tgt_bbox = box_ops.box_cxcywh_to_xyxy(tgt_bbox.unsqueeze(0))
            pred_bbox = box_ops.box_cxcywh_to_xyxy(pred_bbox.unsqueeze(0))

            iou, _ = box_ops.box_iou(tgt_bbox, pred_bbox)
            if iou == 0:
                continue

            img_height, img_width = img_size[0], img_size[1]
            pred_heatmap_scaled = TF.resize(
                pred_heatmap.unsqueeze(0),
                (img_height, img_width),
            ).squeeze()

            tgt_heatmap_scaled = get_multi_hot_map(
                tgt_gaze_point[~tgt_gaze_point_padding],
                (img_height, img_width),
                device=pred_heatmap_scaled.device,
            )

            auc_score = auroc(
                pred_heatmap_scaled.flatten(),
                tgt_heatmap_scaled.flatten(),
                task="binary",
                num_classes=self.gaze_heatmap_size**2,
            )
            self.metric(auc_score.item())
