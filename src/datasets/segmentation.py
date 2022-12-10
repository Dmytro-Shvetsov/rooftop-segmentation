import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple
from albumentations import Compose
from skimage.measure import label

import numpy as np
import cv2
from torch.utils.data import Dataset

from src.utils.io import read_image


class WatershedTiledSegmentationDataset(Dataset):
    """
    PyTorch dataset class that implements logic of parsing and loading stored images data.
    """

    def __init__(self, 
                 root_dir:str, 
                 transforms:Optional[Callable]=None,
                 classes=('roof',),
                 label_codes=None,
                 compute_cc=False,
                 **kwargs):
        self.root_dir = Path(root_dir)

        self.transforms:Compose = transforms
        self.classes = classes
        self.compute_cc = compute_cc
        self.annotation_paths = list(self.root_dir.glob('masks/**/*.npy'))

        if len(classes) > 1:
            # specifiend label_codes and background
            self.label_codes = [(0, 0, 0)] + list(map(tuple, label_codes))
            self.id2code = {k:v for k, v in enumerate(self.label_codes)}
        else:
            self.label_codes = [(255, 255, 255)]

        # assert len(set(p.parent.name for p in self.annotation_paths).difference(instance_names)) == 0

    def __len__(self):
        return len(self.annotation_paths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray]:
        """
        Reads the frame by the index in the found image paths, creates its ground truth masks and applies augmentations if needed.

        Args:
            index (int): index in range [0; len(dataset)]

        Returns:
            Tuple[np.ndarray]: tuple of image (CxHxW), semantic segmentation mask (CxHxW) and watershed energies (CxHxW) numpy arrays.
        """
        mask_path = str(self.annotation_paths[index])
        img_path = mask_path.replace('masks', 'images')
        dataset_name = img_path.split(os.path.sep)[-4]

        if mask_path.endswith('.npy'):
            image = read_image(img_path.replace('.npy', '.png'), npy=True)
            mask = np.load(mask_path)
        else:
            image = read_image(img_path, npy=True)
            mask = read_image(mask_path, npy=True)
        sample = {'image': image, 'mask': mask}

        if self.transforms:
            sample = self.transforms(**sample)

        image, mask = sample['image'], sample['mask']

        if mask.shape[-1] == 4:
            semseg, wngy = mask[..., :3], mask[..., 3]
            semseg = cv2.cvtColor(semseg, cv2.COLOR_RGB2GRAY)
            semseg = (semseg > 0).astype(np.uint8)
        elif mask.shape[-1] == 2:
            semseg, wngy = mask[..., 0], mask[..., 1]
        else:
            raise RuntimeError('Invalid dataset format')

        semseg = np.clip(semseg, 0, 1)
        outs = image, semseg[..., None], wngy[..., None]
        if self.compute_cc:
            outs += (label(semseg.squeeze()),)
        return outs
