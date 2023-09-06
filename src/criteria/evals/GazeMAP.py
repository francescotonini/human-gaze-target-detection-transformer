import torch
from torchmetrics import MeanMetric

from src.utils import box_ops
from src.utils.gaze_ops import get_heatmap_peak_coords, get_l2_dist
from src.utils.misc import get_annotation_id


class GazeMAP:
    def __init__(
        self,
        gaze_heatmap_size: int = 64,
        iou_threshold: int = 0.5,
        l2_threshold: int = 0.15,
    ):
        super().__init__()

        self.gaze_heatmap_size = gaze_heatmap_size
        self.iou_threshold = iou_threshold
        self.l2_threshold = l2_threshold
        self.metric = MeanMetric()

    def reset_metrics(self):
        self.metric.reset()

    def get_metrics(self):
        return {
            "gaze_map": self.metric.compute().item(),
        }

    @torch.no_grad()
    def __call__(self, outputs, targets, indices, **kwargs):
        # If metric is not on the same device as outputs, put it
        # on the same device as outputs
        if self.metric.device != outputs["pred_logits"].device:
            self.metric = self.metric.to(outputs["pred_logits"].device)

        pred_bboxes = outputs["pred_boxes"]
        pred_logits = outputs["pred_logits"]
        pred_gaze_heatmap = outputs["pred_gaze_heatmap"]
        pred_labels = pred_logits.argmax(dim=-1)

        # Target bbox, labels, watch outside and heatmap
        tgt_bboxes = [t["boxes"] for t in targets]
        tgt_labels = [t["labels"] for t in targets]
        tgt_gaze_points = [t["gaze_points"] for t in targets]
        tgt_gaze_points_padding = [t["gaze_points_padding"] for t in targets]
        tgt_regression_padding = [t["regression_padding"].squeeze() for t in targets]
        # Keep only faces (i.e. tgt_regression_padding == False)
        tgt_bboxes = [
            tgt_bboxes[i][~tgt_regression_padding[i]] for i in range(len(tgt_bboxes))
        ]
        tgt_labels = [
            tgt_labels[i][~tgt_regression_padding[i]] for i in range(len(tgt_labels))
        ]
        tgt_gaze_points = [
            tgt_gaze_points[i][~tgt_regression_padding[i]]
            for i in range(len(tgt_gaze_points))
        ]
        tgt_gaze_points_padding = [
            tgt_gaze_points_padding[i][~tgt_regression_padding[i]]
            for i in range(len(tgt_gaze_points_padding))
        ]

        # Assert that all tgt_labels are faces
        assert all(
            [
                tgt_labels[i].all() == get_annotation_id("face")
                for i in range(len(tgt_labels))
            ]
        )

        batch_size = pred_bboxes.shape[0]
        for b_idx in range(batch_size):
            # Keep only faces
            b_pred_labels = pred_labels[b_idx]
            b_pred_faces_presence = b_pred_labels == get_annotation_id("face")
            b_pred_bboxes = pred_bboxes[b_idx][b_pred_faces_presence]
            b_pred_logits = pred_logits[b_idx][b_pred_faces_presence]
            b_pred_gaze_heatmap = pred_gaze_heatmap[b_idx][b_pred_faces_presence]

            # Sort predictions by confidence
            b_sorted_indices = torch.argsort(
                b_pred_logits.max(dim=-1).values, descending=True
            )
            b_pred_bboxes = b_pred_bboxes[b_sorted_indices]
            b_pred_gaze_heatmap = b_pred_gaze_heatmap[b_sorted_indices]

            b_tgt_bboxes = tgt_bboxes[b_idx]
            b_tgt_gaze_points = tgt_gaze_points[b_idx]
            b_tgt_gaze_points_padding = tgt_gaze_points_padding[b_idx]

            num_predictions = b_pred_logits.shape[0]
            num_targets = b_tgt_bboxes.shape[0]
            true_positives = torch.zeros(num_predictions)
            false_positives = torch.zeros(num_predictions)

            for pred_idx in range(num_predictions):
                this_pred_bbox = box_ops.box_cxcywh_to_xyxy(
                    b_pred_bboxes[pred_idx].unsqueeze(0)
                )
                this_pred_gaze_heatmap = b_pred_gaze_heatmap[pred_idx].reshape(
                    self.gaze_heatmap_size, self.gaze_heatmap_size
                )

                best_iou = 0.0
                for tgt_idx in range(num_targets):
                    this_tgt_bbox = box_ops.box_cxcywh_to_xyxy(
                        b_tgt_bboxes[tgt_idx].unsqueeze(0)
                    )
                    this_tgt_gaze_points = b_tgt_gaze_points[tgt_idx]
                    this_tgt_gaze_points_padding = b_tgt_gaze_points_padding[tgt_idx]

                    # Calculate IoU
                    iou, _ = box_ops.box_iou(this_tgt_bbox, this_pred_bbox)
                    if iou < self.iou_threshold:
                        continue

                    # Calculate L2 distance
                    pred_gaze_x, pred_gaze_y = get_heatmap_peak_coords(
                        this_pred_gaze_heatmap
                    )
                    pred_gaze_coord_norm = (
                        torch.tensor(
                            [pred_gaze_x, pred_gaze_y],
                            device=this_pred_gaze_heatmap.device,
                        )
                        / this_pred_gaze_heatmap.shape[
                            0
                        ]  # NOTE: this assumes heatmap is square
                    ).unsqueeze(0)
                    # Average distance: distance between the predicted point and human average point
                    mean_gt_gaze = torch.mean(
                        this_tgt_gaze_points[~this_tgt_gaze_points_padding], 0
                    ).unsqueeze(0)

                    l2_distance = get_l2_dist(mean_gt_gaze, pred_gaze_coord_norm)
                    if l2_distance > self.l2_threshold:
                        continue

                    if iou > best_iou:
                        best_iou = iou

                if best_iou >= self.iou_threshold:
                    # Detection is a true positive
                    true_positives[pred_idx] = 1
                else:
                    # Detection is a false positive
                    false_positives[pred_idx] = 1

            # Compute precision and recall
            cum_true_positives = torch.cumsum(true_positives, dim=0)
            cum_false_positives = torch.cumsum(false_positives, dim=0)

            precision = cum_true_positives / (cum_true_positives + cum_false_positives)
            recall = cum_true_positives / num_targets

            # Calculate average precision using the area under the precision-recall curve
            ap = torch.tensor(0.0, device=pred_logits.device)
            for t in torch.linspace(0, 1, 11):
                mask = recall >= t
                if mask.any():
                    ap += torch.max(precision[mask])

            # Calculate mean average precision (mAP)
            self.metric(ap / 11)
