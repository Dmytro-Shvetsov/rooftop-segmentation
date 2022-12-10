from os import PathLike
from typing import Union

import cv2
import numpy as np
from PIL import Image

from .base import BaseVisualizer


class ImagesGridVisualizer(BaseVisualizer):
    def __init__(self, width, height, cols, sep_size_px):
        self._width = width
        self._height = height
        self._cols = cols
        self._sep_size_px = sep_size_px
        self._img = None
    
    def _make_row(self, images, target_size):
        if len(images) == 0:
            return np.array([])
        n_images = len(images)
        width, height = target_size
        sep = np.ones((height, self._sep_size_px, 3), dtype=np.uint8) * 255
        item_width = width / len(images)
        item_width -= (n_images - 1) * (self._sep_size_px / n_images)
        item_width = int(item_width)

        ret = []
        for img in images:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, (item_width, height), interpolation=cv2.INTER_LINEAR)
            ret.append(img)
            ret.append(sep)
        
        ret.pop()
        return np.hstack(ret)

    def make_grid(self, images):
        sep = np.ones((self._sep_size_px, self._width, 3), dtype=np.uint8) * 255
        
        n_images = len(images)

        item_height = self._height / self._cols
        item_height -= (n_images - 1) * (self._sep_size_px / self._cols)
        item_height = int(item_height)
        
        rows = []
        for i in range(0, n_images, self._cols):
            imgs = images[i:i + self._cols]
            row = self._make_row(imgs, (self._width, item_height))
            sep = np.ones((self._sep_size_px, row.shape[1], 3), dtype=np.uint8) * 255
            rows.append(row)
            rows.append(sep)
        rows.pop()
        return np.vstack(rows)

    def draw(self, images):
        self._img = self.make_grid(images)

    def get_image(self, pil=False) -> Union[np.ndarray, Image.Image]:
        return Image.fromarray(self._img) if pil else self._img

    def save(self, file_path: PathLike) -> None:
        if self._img is None:
            return
        Image.fromarray(self._img).save(file_path)

