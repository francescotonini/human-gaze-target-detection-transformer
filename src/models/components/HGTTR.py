import torch
import torch.nn as nn
from torch.utils import model_zoo
from torch_intermediate_layer_getter import IntermediateLayerGetter

from libs.detr.models import build_model as build_detr
from src import utils
from src.models.components.MLP import MLP
from src.utils.AttributeDict import AttributeDict
from src.utils.misc import load_pretrained

log = utils.get_pylogger(__name__)


class HGTTR(nn.Module):
    def __init__(
        self,
        num_queries: int,
        gaze_heatmap_size: int,
        aux_loss: bool,
    ):
        super().__init__()

        self.backbone, _, _ = build_detr(
            AttributeDict(
                {
                    "dataset_file": "coco",
                    "device": "cuda",  # TODO: override this using hydra config
                    "num_queries": num_queries,
                    "aux_loss": False,  # We use our own aux loss
                    "masks": False,
                    "eos_coef": 0,
                    "hidden_dim": 256,
                    "position_embedding": "sine",
                    "lr_backbone": 1e-5,
                    "backbone": "resnet50",
                    "dilation": False,
                    "nheads": 8,
                    "dim_feedforward": 2048,
                    "enc_layers": 6,
                    "dec_layers": 6,
                    "dropout": 0.1,
                    "pre_norm": False,
                    "set_cost_class": 1,
                    "set_cost_bbox": 5,
                    "set_cost_giou": 2,
                    "mask_loss_coef": 1,
                    "bbox_loss_coef": 5,
                    "giou_loss_coef": 2,
                }
            )
        )
        load_pretrained(
            self.backbone,
            model_zoo.load_url(
                "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"
            ),
        )

        self.backbone_getter = IntermediateLayerGetter(
            self.backbone,
            return_layers={
                "transformer": "hs",
            },
        )

        self.aux_loss = aux_loss
        backbone_hidden_dim = self.backbone.transformer.d_model

        self.class_embed = MLP(backbone_hidden_dim, backbone_hidden_dim, 2, 1)
        self.gaze_watch_outside_embed = self.watch_outside_embed = MLP(
            backbone_hidden_dim, backbone_hidden_dim, 1, 1
        )
        self.head_bbox_embed = MLP(backbone_hidden_dim, backbone_hidden_dim, 4, 3)
        self.gaze_heatmap_embed = MLP(
            backbone_hidden_dim,
            256,
            gaze_heatmap_size**2,
            5,
        )

    def forward(self, samples):
        intermediate_layers, _ = self.backbone_getter(samples)
        decoder_features = intermediate_layers["hs"][0]

        outputs_logits = self.class_embed(decoder_features)
        outputs_bbox = self.head_bbox_embed(decoder_features).sigmoid()
        # TODO: sigmoid?
        outputs_gaze_watch_outside = self.gaze_watch_outside_embed(decoder_features)
        outputs_gaze_heatmap = self.gaze_heatmap_embed(decoder_features).sigmoid()

        out = {
            "pred_logits": outputs_logits[-1],
            "pred_boxes": outputs_bbox[-1],
            "pred_gaze_watch_outside": outputs_gaze_watch_outside[-1],
            "pred_gaze_heatmap": outputs_gaze_heatmap[-1],
        }
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_logits,
                outputs_bbox,
                outputs_gaze_watch_outside,
                outputs_gaze_heatmap,
            )

        return out

    @torch.jit.unused
    def _set_aux_loss(
        self, outputs_class, outputs_coord, outputs_watch_outside, outputs_gaze_heatmap
    ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {
                "pred_logits": a,
                "pred_boxes": b,
                "pred_gaze_watch_outside": c,
                "pred_gaze_heatmap": d,
            }
            for a, b, c, d in zip(
                outputs_class[:-1],
                outputs_coord[:-1],
                outputs_watch_outside[:-1],
                outputs_gaze_heatmap[:-1],
            )
        ]
