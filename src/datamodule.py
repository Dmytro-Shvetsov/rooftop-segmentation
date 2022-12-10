from collections import OrderedDict
from pathlib import Path
from pprint import pprint
from typing import Dict, Optional, Tuple
import pytorch_lightning as pl
import albumentations as albu

from torch.utils.data import DataLoader

from src.datasets.segmentation import WatershedTiledSegmentationDataset


class SegmentationDataModule(pl.LightningDataModule):
    """PyTorch Lightning datamodule"""

    def __init__(self, config):
        super().__init__()
        self.cfg = config

        self.root_dir = Path(self.cfg.dataset_dir)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def get_augmentations(self, aux_transforms:Optional[Dict]=None) -> albu.Compose:
        """
        Creates augmentations callable from serialized dictionary.

        Args:
            target_size (Tuple[int]): target images resolution (hxw)
            aux_transforms (Optional[albu.Compose], optional): dictionary defining augmentation operations. Defaults to None.

        Returns:
            albu.Compose: final augmentations object
        """
        transforms = []
        if aux_transforms is not None:
            aux_transforms = albu.from_dict(aux_transforms)
            transforms.append(aux_transforms)
        # transforms.append(albu.Resize(*target_size, interpolation=cv2.INTER_LINEAR))
        return albu.Compose(transforms)

    def prepare_data(self):
        """Initializes train/validation/test datasets."""
        # download data
        train_transforms = self.get_augmentations(self.cfg.get('train_aug'))
        self.train_dataset = WatershedTiledSegmentationDataset(self.root_dir / 'train', train_transforms, self.cfg.classes, self.cfg.get('label_codes'))

        val_transforms = self.get_augmentations(self.cfg.get('val_aug'))
        self.val_dataset = WatershedTiledSegmentationDataset(self.root_dir / 'val', val_transforms, self.cfg.classes, self.cfg.get('label_codes'))

        self.test_dataset = WatershedTiledSegmentationDataset(self.root_dir / 'test', val_transforms, self.cfg.classes, self.cfg.get('label_codes'))

        print(f'Num tiles per split(train/val/test): {len(self.train_dataset)}/{len(self.val_dataset)}/{len(self.test_dataset)}')

    def train_dataloader(self):
        """Creates the training DataLoader object for batches sampling."""
        return DataLoader(self.train_dataset, 
                          batch_size=self.cfg.batch_size, 
                          num_workers=self.cfg.num_workers, 
                          shuffle=True, 
                          drop_last=True)

    def val_dataloader(self):
        """Creates the validation DataLoader object for batches sampling."""
        return DataLoader(self.val_dataset, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers)

    def test_dataloader(self):
        """For now, returns the validation DataLoader as the testing one. It will be changed if the test dataset will be created."""
        return DataLoader(self.test_dataset, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers)
