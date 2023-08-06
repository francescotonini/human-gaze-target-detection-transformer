from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

import src.datamodules.components.transforms as T
from src.datamodules.components.GazeFollow import GazeFollow
from src.utils import misc


class GazeFollowDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        num_queries: int = 20,
        gaze_heatmap_size: int = 64,
        gaze_heatmap_default_value: float = 0.0,
        use_aux_heads_dataset: bool = False,
        use_gaze_inside_only: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.transforms_train: T.Compose = self._get_transforms(is_train=True)
        self.transforms_val: T.Compose = self._get_transforms(is_train=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    @staticmethod
    def _get_transforms(is_train: bool):
        normalize = T.Compose(
            [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        if is_train:
            img_transform = T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.RandomSelect(
                        T.RandomResize(
                            [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
                            max_size=1333,
                        ),
                        T.Compose(
                            [
                                T.RandomResize([400, 500, 600]),
                                T.RandomCrop(p=0.5),
                                T.RandomResize(
                                    [
                                        480,
                                        512,
                                        544,
                                        576,
                                        608,
                                        640,
                                        672,
                                        704,
                                        736,
                                        768,
                                        800,
                                    ],
                                    max_size=1333,
                                ),
                            ]
                        ),
                    ),
                    normalize,
                ]
            )
        else:
            img_transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    normalize,
                ]
            )

        return img_transform

    def setup(self, stage: Optional[str] = None):
        # Load and split datasets only if not loaded already
        if self.data_train or self.data_val:
            return

        self.data_train = GazeFollow(
            self.hparams.data_dir,
            self.transforms_train,
            is_train=True,
            num_queries=self.hparams.num_queries,
            gaze_heatmap_size=self.hparams.gaze_heatmap_size,
            gaze_heatmap_default_value=self.hparams.gaze_heatmap_default_value,
            use_aux_heads_dataset=self.hparams.use_aux_heads_dataset,
            use_gaze_inside_only=self.hparams.use_gaze_inside_only,
        )
        self.data_val = GazeFollow(
            self.hparams.data_dir,
            self.transforms_val,
            is_train=False,
            num_queries=self.hparams.num_queries,
            gaze_heatmap_size=self.hparams.gaze_heatmap_size,
            gaze_heatmap_default_value=self.hparams.gaze_heatmap_default_value,
            use_aux_heads_dataset=self.hparams.use_aux_heads_dataset,
            use_gaze_inside_only=self.hparams.use_gaze_inside_only,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=True,
            collate_fn=misc.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
            collate_fn=misc.collate_fn,
        )


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "gazefollow.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
