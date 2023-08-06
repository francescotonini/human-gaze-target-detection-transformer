from typing import Any, List

import pytorch_lightning
import torch
import torchvision.transforms.functional as TF
import wandb
from matplotlib import pyplot as plt, patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import LightningModule
from torch.utils import model_zoo
from torchmetrics import MeanMetric

from src.utils.box_ops import box_cxcywh_to_xyxy
from src.utils.gaze_ops import get_heatmap_peak_coords
from src.utils.misc import (
    get_annotations,
    get_annotation_id,
    unnorm,
    fig2img,
    load_pretrained,
)
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class HGTTRLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: torch.nn.Module,
        evaluation: torch.nn.Module,
        net_pretraining: str = None,
        n_of_images_to_log: int = 0,
    ):
        super().__init__()

        # This line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False, ignore=["net", "criterion", "evaluation"]
        )

        self.net = net
        self.criterion = criterion
        self.evaluation = evaluation

        if net_pretraining is not None:
            log.info(f"Loading pretrained model from {net_pretraining}")

            load_pretrained(self.net, model_zoo.load_url(net_pretraining))

        # For averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def on_start(self):
        # Clean caches
        torch.cuda.empty_cache()
        self.evaluation.reset()

    def on_train_start(self):
        self.on_start()

    def on_validation_start(self):
        self.on_start()

    def step(self, batch: Any, do_eval: bool = False):
        samples, targets = batch
        outputs = self.net(samples)

        losses = self.criterion(outputs, targets)
        if do_eval:
            self.evaluation(outputs, targets)

        return sum(losses.values()), losses, outputs

    def training_step(self, batch: Any, batch_idx: int):
        total_loss, losses, _ = self.step(batch)

        self.train_loss(total_loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=batch[0].tensors.shape[0],
        )
        self.log(
            "train/lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=batch[0].tensors.shape[0],
        )
        self.log(
            "train/lr_backbone",
            self.trainer.optimizers[0].param_groups[1]["lr"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=batch[0].tensors.shape[0],
        )

        # Log each loss
        for loss_name, loss in losses.items():
            self.log(
                f"train/{loss_name}",
                loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                batch_size=batch[0].tensors.shape[0],
            )

        return {"loss": total_loss}

    def validation_step(self, batch: Any, batch_idx: int):
        total_loss, losses, outputs = self.step(batch, do_eval=True)
        bs = batch[0].tensors.shape[0]

        self.val_loss(total_loss)
        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=bs,
        )

        # Log each loss
        for loss_name, loss in losses.items():
            self.log(
                f"val/{loss_name}",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=bs,
            )

        # Estimate how many images to log from this batch and batch_size
        n_of_images_logged_so_far = bs * batch_idx
        n_of_images_to_log = self.hparams.n_of_images_to_log
        n_of_images_still_to_log = n_of_images_to_log - n_of_images_logged_so_far
        log_images = n_of_images_still_to_log > 0

        return {
            "loss": total_loss,
            "batch": batch if log_images else None,
            "outputs": outputs if log_images else None,
            "n_of_images_still_to_log": n_of_images_still_to_log,
        }

    def validation_step_end(self, outputs: List[Any]):
        if outputs["batch"] is None:
            return

        n_of_images_still_to_log = outputs["n_of_images_still_to_log"]

        # Get "batch" from the first element of outputs
        batch = outputs["batch"]
        samples, targets = batch
        outputs = outputs["outputs"]

        # Stop here if logger is not WandbLogger
        if not isinstance(self.logger, pytorch_lightning.loggers.WandbLogger):
            log.warning("Logger is not WandbLogger, skipping logging to wandb")

            return

        # Log images with bounding boxes on wandb
        wandb_od_images = []
        for idx in range(min(n_of_images_still_to_log, len(samples.tensors))):
            sample = samples.tensors[idx]
            mask = samples.mask[idx]

            # Remove mask from sample
            sample = sample[:, ~mask[:, 0], :]
            sample = sample[:, :, ~mask[0, :]]

            target_box_data = []
            target_boxes = targets[idx]["boxes"]
            target_labels = targets[idx]["labels"]
            for box, label in zip(target_boxes, target_labels):
                box = box_cxcywh_to_xyxy(box)
                target_box_data.append(
                    {
                        "position": {
                            "minX": box[0].item(),
                            "minY": box[1].item(),
                            "maxX": box[2].item(),
                            "maxY": box[3].item(),
                        },
                        "class_id": label.argmax().item(),
                        "scores": {"prob": label.max().item()},
                    }
                )

            pred_box_data = []
            pred_boxes = outputs["pred_boxes"][idx]
            pred_labels = outputs["pred_logits"][idx]
            for box, label in zip(pred_boxes, pred_labels):
                box = box_cxcywh_to_xyxy(box)
                pred_box_data.append(
                    {
                        "position": {
                            "minX": box[0].item(),
                            "minY": box[1].item(),
                            "maxX": box[2].item(),
                            "maxY": box[3].item(),
                        },
                        "class_id": label.argmax().item(),
                        "scores": {"prob": label.softmax(-1).max().item()},
                    }
                )

            img = wandb.Image(
                sample,
                boxes={
                    "predictions": {
                        "box_data": pred_box_data,
                        "class_labels": get_annotations(),
                    },
                    "targets": {
                        "box_data": target_box_data,
                        "class_labels": get_annotations(),
                    },
                },
            )

            wandb_od_images.append(img)

        # Log images with gaze heatmaps on wandb
        wandb_gaze_heatmap_images = []
        for idx in range(min(n_of_images_still_to_log, len(samples.tensors))):
            sample = samples.tensors[idx]
            mask = samples.mask[idx]

            # Remove mask from sample
            sample = sample[:, ~mask[:, 0], :]
            sample = sample[:, :, ~mask[0, :]]

            pred_labels = outputs["pred_logits"][idx].argmax(-1)
            face_idxs = (
                (pred_labels == get_annotation_id("face", face_only=True))
                .nonzero()
                .flatten()
                .tolist()
            )

            for face_idx in face_idxs:
                head_box = (
                    box_cxcywh_to_xyxy(outputs["pred_boxes"][idx][face_idx]).cpu()
                    * torch.tensor([sample.shape[2], sample.shape[1]] * 2)
                ).numpy()
                gaze_heatmap = (
                    outputs["pred_gaze_heatmap"][idx][face_idx].reshape(64, 64).cpu()
                )

                gaze_heatmap = (
                    TF.resize(
                        gaze_heatmap.unsqueeze(0),  # Add channel dim
                        (sample.shape[1], sample.shape[2]),  # [h, w]
                    ).squeeze(0)
                ).numpy()

                pred_gaze_x, pred_gaze_y = torch.tensor(
                    get_heatmap_peak_coords(gaze_heatmap)
                )

                fig, (ax_bbox, ax_heatmap, ax_heatmap_unscaled,) = plt.subplots(
                    1,
                    3,
                    figsize=((sample.shape[2] * 3) / 96, sample.shape[1] / 96),
                    dpi=96,
                )

                ax_bbox.axis("off")
                ax_bbox.imshow(
                    unnorm(sample).permute(1, 2, 0).cpu().numpy(), vmin=0, vmax=1
                )

                # Head bbox
                ax_bbox.add_patch(
                    patches.Rectangle(
                        (head_box[0], head_box[1]),
                        head_box[2] - head_box[0],
                        head_box[3] - head_box[1],
                        linewidth=2,
                        edgecolor=(1, 0, 0),
                        facecolor="none",
                    )
                )
                # Arrow from center of head bbox to gaze point
                ax_bbox.arrow(
                    (head_box[0] + head_box[2]) / 2,
                    (head_box[1] + head_box[3]) / 2,
                    pred_gaze_x - (head_box[0] + head_box[2]) / 2,
                    pred_gaze_y - (head_box[1] + head_box[3]) / 2,
                    head_width=10,
                    head_length=10,
                    width=0.1,
                    color="r",
                )
                ax_bbox.set_title(
                    "Predicted head bbox (gaze source) and gaze point (heatmap peak)"
                )

                ax_heatmap.axis("off")
                ax_heatmap.imshow(
                    unnorm(sample).permute(1, 2, 0).cpu().numpy(), vmin=0, vmax=1
                )
                im = ax_heatmap.imshow(
                    gaze_heatmap, cmap="jet", alpha=0.3, vmin=0, vmax=1
                )
                divider = make_axes_locatable(ax_heatmap)
                cax = divider.append_axes("right", size="3%", pad=0.05)
                plt.colorbar(im, cax=cax)
                ax_heatmap.set_title("Gaze heatmap normalized [0, 1]")

                ax_heatmap_unscaled.axis("off")
                ax_heatmap_unscaled.imshow(
                    unnorm(sample).permute(1, 2, 0).cpu().numpy(), vmin=0, vmax=1
                )
                im = ax_heatmap_unscaled.imshow(gaze_heatmap, cmap="jet", alpha=0.3)
                divider = make_axes_locatable(ax_heatmap_unscaled)
                cax = divider.append_axes("right", size="3%", pad=0.05)
                plt.colorbar(im, cax=cax)
                ax_heatmap_unscaled.set_title("Gaze heatmap unscaled")

                plt.suptitle(f"Sample {idx} of {len(samples.tensors)}")

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig = plt.gcf()

                final_image = fig2img(fig)

                # Cleanup matplotlib
                plt.close(fig)
                plt.close("all")
                del fig

                wandb_gaze_heatmap_images.append(wandb.Image(final_image))

        self.logger.experiment.log(
            {
                "val/images": wandb_od_images,
                "val/gaze_heatmap_images": wandb_gaze_heatmap_images,
                "trainer/global_step": self.trainer.global_step,
            }
        )

    def on_validation_epoch_end(self):
        # Log each eval
        for eval_name, eval_value in self.evaluation.get_metrics().items():
            self.log(
                f"val/{eval_name}",
                eval_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

    def configure_optimizers(self):
        params = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone.backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone.backbone" in n and p.requires_grad
                ],
                "lr": self.hparams.optimizer.keywords["lr"] / 10,
            },
        ]

        optimizer = self.hparams.optimizer(params=params)

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return [optimizer], [scheduler]

        return optimizer


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "hgttr.yaml")
    _ = hydra.utils.instantiate(cfg)
