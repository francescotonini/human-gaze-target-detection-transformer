import os
from typing import Any, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import BoolTensor, FloatTensor
from torch.utils.data.dataset import Dataset
from torchvision.ops import box_iou
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
)

import src.datamodules.components.transforms as T
import src.utils as utils
from src.utils.box_ops import box_xyxy_to_cxcywh
from src.utils.gaze_ops import get_label_map
from src.utils.misc import get_annotation_id

log = utils.get_pylogger(__name__)


class GazeFollow(Dataset):
    def __init__(
        self,
        data_dir: str,
        transforms: T.Compose,
        is_train: bool = True,
        num_queries: int = 20,
        gaze_heatmap_size: int = 64,
        gaze_heatmap_default_value: float = 0.0,
        use_aux_heads_dataset: bool = False,
        use_gaze_inside_only: bool = False,
        bbox_overflow_coeff: float = 0.1,
        min_aux_faces_score: float = 0.9,
    ):
        self.data_dir = data_dir
        self.transforms = transforms
        self.is_train_set = is_train
        self.num_queries = num_queries
        self.gaze_heatmap_default_value = gaze_heatmap_default_value
        self.gaze_heatmap_size = gaze_heatmap_size
        self.use_aux_heads_dataset = use_aux_heads_dataset
        # Will increase/decrease the bbox of objects by this value (%)
        self.bbox_overflow_coeff = bbox_overflow_coeff
        self.min_aux_faces_score = min_aux_faces_score

        self._prepare_dataset(use_gaze_inside_only)

        if self.use_aux_heads_dataset:
            self._prepare_aux_dataset()

    def _prepare_dataset(self, use_gaze_inside_only: bool):
        labels_path = os.path.join(
            self.data_dir,
            "train_annotations_release.txt"
            if self.is_train_set
            else "test_annotations_release.txt",
        )

        column_names = [
            "path",
            "idx",
            "body_bbox_x",
            "body_bbox_y",
            "body_bbox_w",
            "body_bbox_h",
            "eye_x",
            "eye_y",
            "gaze_x",
            "gaze_y",
            "bbox_x_min",
            "bbox_y_min",
            "bbox_x_max",
            "bbox_y_max",
        ]
        if self.is_train_set:
            column_names.append("inout")
        column_names.append("orig_dataset")

        df = pd.read_csv(
            labels_path,
            sep=",",
            names=column_names,
            usecols=column_names,
            index_col=False,
        )

        # Only use "in" or "out "gaze (-1 is invalid, 0 is out gaze)
        if self.is_train_set:
            if use_gaze_inside_only:
                df = df[df["inout"] == 1]
            else:
                df = df[df["inout"] != -1]

        # Drop rows with invalid bboxes
        coords = torch.tensor(
            np.array(
                (
                    df["bbox_x_min"].values,
                    df["bbox_y_min"].values,
                    df["bbox_x_max"].values,
                    df["bbox_y_max"].values,
                )
            ).transpose(1, 0)
        )
        valid_bboxes = (coords[:, 2:] >= coords[:, :2]).all(dim=1)

        df = df.loc[valid_bboxes.tolist(), :]
        df.reset_index(inplace=True)
        df = df.groupby("path")

        self.keys = list(df.groups.keys())
        self.X = df
        self.length = len(self.keys)

    def _prepare_aux_dataset(self):
        labels_path = os.path.join(
            self.data_dir,
            "train_heads.csv" if self.is_train_set else "test_heads.csv",
        )

        column_names = [
            "path",
            "score",
            "head_bbox_x_min",
            "head_bbox_y_min",
            "head_bbox_x_max",
            "head_bbox_y_max",
        ]

        df = pd.read_csv(
            labels_path,
            sep=",",
            names=column_names,
            usecols=column_names,
            skiprows=[
                0,
            ],
            index_col=False,
        )

        # Keep only heads with high score
        df = df[df["score"] >= self.min_aux_faces_score]

        # Drop rows with invalid bboxes
        coords = torch.tensor(
            np.array(
                (
                    df["head_bbox_x_min"].values,
                    df["head_bbox_y_min"].values,
                    df["head_bbox_x_max"].values,
                    df["head_bbox_y_max"].values,
                )
            ).transpose(1, 0)
        )
        valid_bboxes = (coords[:, 2:] >= coords[:, :2]).any(dim=1)
        df = df.loc[valid_bboxes.tolist(), :]

        df.reset_index(inplace=True)
        df = df.groupby("path")

        self.X_aux = df
        self.aux_keys = list(df.groups.keys())

    def __getitem__(
        self, index: int
    ) -> tuple[Any, dict[str, Union[FloatTensor, BoolTensor]]]:
        if self.is_train_set:
            return self.__get_train_item__(index)
        else:
            return self.__get_test_item__(index)

    def __len__(self) -> int:
        return self.length

    def __get_train_item__(
        self, index: int
    ) -> tuple[Any, dict[str, Union[FloatTensor, BoolTensor]]]:
        # Load image
        img = Image.open(os.path.join(self.data_dir, self.keys[index]))
        img = img.convert("RGB")
        img_width, img_height = img.size

        boxes = []
        gaze_points = []
        gaze_heatmaps = []
        gaze_watch_outside = []
        for _, row in self.X.get_group(self.keys[index]).iterrows():
            # Load bbox
            box_x_min = row["bbox_x_min"]
            box_y_min = row["bbox_y_min"]
            box_x_max = row["bbox_x_max"]
            box_y_max = row["bbox_y_max"]

            # Expand bbox
            box_width = box_x_max - box_x_min
            box_height = box_y_max - box_y_min
            box_x_min -= box_width * self.bbox_overflow_coeff
            box_y_min -= box_height * self.bbox_overflow_coeff
            box_x_max += box_width * self.bbox_overflow_coeff
            box_y_max += box_height * self.bbox_overflow_coeff

            # Jitter
            if np.random.random_sample() <= 0.5:
                bbox_overflow_coeff = np.random.random_sample() * 0.2
                box_x_min -= box_width * bbox_overflow_coeff
                box_y_min -= box_height * bbox_overflow_coeff
                box_x_max += box_width * bbox_overflow_coeff
                box_y_max += box_height * bbox_overflow_coeff

            boxes.append(
                torch.FloatTensor([box_x_min, box_y_min, box_x_max, box_y_max])
            )

            # Gaze point
            gaze_x = row["gaze_x"] * img_width
            gaze_y = row["gaze_y"] * img_height
            gaze_points.append(torch.FloatTensor([gaze_x, gaze_y]).view(1, 2))

            # Gaze watch outside
            gaze_watch_outside.append(row["inout"] == 0)

        if self.use_aux_heads_dataset and self.keys[index] in self.aux_keys:
            aux_head_boxes = []
            for _, row in self.X_aux.get_group(self.keys[index]).iterrows():
                # Head bbox
                box_x_min = row["head_bbox_x_min"]
                box_y_min = row["head_bbox_y_min"]
                box_x_max = row["head_bbox_x_max"]
                box_y_max = row["head_bbox_y_max"]

                box_width = box_x_max - box_x_min
                box_height = box_y_max - box_y_min
                box_x_min -= box_width * self.bbox_overflow_coeff
                box_y_min -= box_height * self.bbox_overflow_coeff
                box_x_max += box_width * self.bbox_overflow_coeff
                box_y_max += box_height * self.bbox_overflow_coeff

                aux_head_boxes.append(
                    torch.FloatTensor([box_x_min, box_y_min, box_x_max, box_y_max])
                )

            # Calculate iou between head_boxes and aux_head_boxes and remove from aux_head_boxes
            # the boxes where iou is not zero
            iou = box_iou(
                torch.stack(boxes),
                torch.stack(aux_head_boxes),
            )
            aux_head_boxes = [
                aux_head_boxes[i]
                for i in range(len(aux_head_boxes))
                if iou[:, i].max() == 0
            ]

            for i in range(min(len(aux_head_boxes), self.num_queries - len(boxes))):
                boxes.append(aux_head_boxes[i])

        # Random color change
        if np.random.random_sample() <= 0.5:
            img = adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
            img = adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
            img = adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

        target = {
            "path": self.keys[index],
            "img_size": torch.FloatTensor([img_height, img_width]),
            "boxes": torch.stack(boxes),
            "gaze_points": torch.stack(gaze_points),
            "gaze_watch_outside": torch.BoolTensor(gaze_watch_outside).long(),
        }

        # Transform image and rescale all bounding target
        img, target = self.transforms(img, target)

        img_size = target["img_size"].repeat(self.num_queries, 1)
        target["img_size"] = img_size

        boxes = torch.full((self.num_queries, 4), 0).float()
        boxes[: len(target["boxes"]), :] = target["boxes"]
        boxes[len(target["boxes"]) :, :] = box_xyxy_to_cxcywh(
            torch.tensor([0, 0, 1, 1])
        )
        labels = torch.full((self.num_queries, 2), 0).float()
        labels[: len(target["boxes"]), get_annotation_id("face", face_only=True)] = 1
        labels[
            len(target["boxes"]) :, get_annotation_id("no-object", face_only=True)
        ] = 1

        target["boxes"] = boxes
        target["labels"] = labels

        # Represents which objects (i.e. heads) have a gaze heatmap
        regression_padding = torch.full((self.num_queries, 1), True)
        regression_padding[: len(target["gaze_points"])] = False
        target["regression_padding"] = regression_padding

        gaze_points = torch.full((self.num_queries, 1, 2), 0).float()
        gaze_points[: len(target["gaze_points"]), :, :] = target["gaze_points"]
        target["gaze_points"] = gaze_points

        gaze_watch_outside = torch.full((self.num_queries, 1), 0).float()
        gaze_watch_outside[: len(target["gaze_watch_outside"]), 0] = target[
            "gaze_watch_outside"
        ]
        target["gaze_watch_outside"] = gaze_watch_outside.long()

        for gaze_point, regression_padding in zip(
            target["gaze_points"], target["regression_padding"]
        ):
            gaze_x, gaze_y = gaze_point.squeeze(0)
            if not regression_padding:
                gaze_heatmap = torch.zeros(
                    (self.gaze_heatmap_size, self.gaze_heatmap_size)
                )

                gaze_heatmap = get_label_map(
                    gaze_heatmap,
                    [
                        gaze_x * self.gaze_heatmap_size,
                        gaze_y * self.gaze_heatmap_size,
                    ],
                    3,
                    pdf="Gaussian",
                )
            else:
                gaze_heatmap = torch.full(
                    (self.gaze_heatmap_size, self.gaze_heatmap_size),
                    float(self.gaze_heatmap_default_value),
                )

            gaze_heatmaps.append(gaze_heatmap)

        target["gaze_heatmaps"] = torch.stack(gaze_heatmaps)

        return img, target

    def __get_test_item__(
        self, index: int
    ) -> tuple[Any, dict[str, Union[FloatTensor, BoolTensor]]]:
        # Load image
        img = Image.open(os.path.join(self.data_dir, self.keys[index]))
        img = img.convert("RGB")
        img_width, img_height = img.size

        boxes = []
        gaze_points = []
        gaze_points_padding = []
        gaze_heatmaps = []
        gaze_watch_outside = []

        # Group annotations from same scene with same person
        for _, same_person_annotations in self.X.get_group(self.keys[index]).groupby(
            "eye_x"
        ):
            # Group annotations of the same person
            sp_gaze_points = []
            sp_boxes = []
            sp_gaze_outside = []
            for _, row in same_person_annotations.iterrows():
                # Load bbox
                box_x_min = row["bbox_x_min"]
                box_y_min = row["bbox_y_min"]
                box_x_max = row["bbox_x_max"]
                box_y_max = row["bbox_y_max"]

                sp_boxes.append(
                    torch.FloatTensor([box_x_min, box_y_min, box_x_max, box_y_max])
                )

                gaze_x = row["gaze_x"] * img_width
                gaze_y = row["gaze_y"] * img_height
                sp_gaze_points.append(torch.FloatTensor([gaze_x, gaze_y]))
                sp_gaze_outside.append(False)

            boxes.append(torch.FloatTensor(sp_boxes[-1]))

            sp_gaze_points_padded = torch.full((20, 2), -1).float()
            sp_gaze_points_padded[: len(sp_gaze_points), :] = torch.stack(
                sp_gaze_points
            )
            sp_gaze_points_padding = torch.full((20,), False)
            sp_gaze_points_padding[len(sp_gaze_points) :] = True

            gaze_points.append(sp_gaze_points_padded)
            gaze_points_padding.append(sp_gaze_points_padding)

            gaze_watch_outside.append(
                (
                    torch.BoolTensor(sp_gaze_outside).sum() > len(sp_gaze_outside) / 2
                ).item()
            )

        target = {
            "path": self.keys[index],
            "img_size": torch.FloatTensor([img_height, img_width]),
            "boxes": torch.stack(boxes),
            "gaze_points": torch.stack(gaze_points),
            "gaze_points_padding": torch.stack(gaze_points_padding),
            "gaze_watch_outside": torch.BoolTensor(gaze_watch_outside).long(),
        }

        # Transform image and rescale all bounding target
        img, target = self.transforms(
            img,
            target,
        )

        img_size = target["img_size"].repeat(self.num_queries, 1)
        target["img_size"] = img_size

        boxes = torch.full((self.num_queries, 4), 0).float()
        boxes[: len(target["boxes"]), :] = target["boxes"]
        boxes[len(target["boxes"]) :, :] = box_xyxy_to_cxcywh(
            torch.tensor([0, 0, 1, 1])
        )
        labels = torch.full((self.num_queries, 2), 0).float()
        labels[: len(target["boxes"]), get_annotation_id("face", face_only=True)] = 1
        labels[
            len(target["boxes"]) :, get_annotation_id("no-object", face_only=True)
        ] = 1

        target["boxes"] = boxes
        target["labels"] = labels

        regression_padding = torch.full((self.num_queries, 1), True)
        regression_padding[: len(target["gaze_points"])] = False
        target["regression_padding"] = regression_padding

        gaze_points = torch.full((self.num_queries, 20, 2), 0).float()
        gaze_points[: len(target["gaze_points"]), :, :] = target["gaze_points"]
        target["gaze_points"] = gaze_points

        gaze_points_padding = torch.full((self.num_queries, 20), False)
        gaze_points_padding[: len(target["gaze_points_padding"]), :] = target[
            "gaze_points_padding"
        ]
        target["gaze_points_padding"] = gaze_points_padding

        gaze_watch_outside = torch.full((self.num_queries, 1), 0).float()
        gaze_watch_outside[: len(gaze_watch_outside), 0] = target["gaze_watch_outside"]
        target["gaze_watch_outside"] = gaze_watch_outside.long()

        for gaze_point, gaze_point_padding, regression_padding in zip(
            target["gaze_points"],
            target["gaze_points_padding"],
            target["regression_padding"],
        ):
            if not regression_padding:
                gaze_heatmap = []

                for (gaze_x, gaze_y), gaze_padding in zip(
                    gaze_point, gaze_point_padding
                ):
                    if gaze_x == -1 or gaze_padding:
                        continue

                    gaze_heatmap.append(
                        get_label_map(
                            torch.zeros(
                                (self.gaze_heatmap_size, self.gaze_heatmap_size)
                            ),
                            [
                                gaze_x * self.gaze_heatmap_size,
                                gaze_y * self.gaze_heatmap_size,
                            ],
                            3,
                            pdf="Gaussian",
                        )
                    )

                gaze_heatmap = torch.stack(gaze_heatmap)
                gaze_heatmap = gaze_heatmap.sum(dim=0) / gaze_heatmap.sum(dim=0).max()
            else:
                gaze_heatmap = torch.full(
                    (self.gaze_heatmap_size, self.gaze_heatmap_size),
                    self.gaze_heatmap_default_value,
                )

            gaze_heatmaps.append(gaze_heatmap)

        target["gaze_heatmaps"] = torch.stack(gaze_heatmaps)

        return img, target
