from pathlib import Path
from typing import Any, Optional

from PIL import Image
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.visualization.images_grid import ImagesGridVisualizer
from src.models.base import SegmentationModelInterface


class DWTVisualizationCallback(Callback):
    def __init__(self, cfg, vis_frequency, n_vis_images=3, image_size=(960, 600), tile_sep=2, mode=None, save_dir='./vis', **kwargs):
        self._cfg = cfg
        self._mode = mode
        self._save_dir = Path(save_dir).resolve()
        self._save_dir.mkdir(exist_ok=True, parents=True)
        self._freq = vis_frequency
        self._n_vis_images = n_vis_images

        grid_size = 4
        self._visualizer = ImagesGridVisualizer(*image_size, grid_size, tile_sep)

    def _build_vis_image(self, batch:Any, model: SegmentationModelInterface, first_n:bool=False, log_key=None) -> Any:
        if first_n:
            batch = [item[:self._n_vis_images] for item in batch]
        else:
            inds = np.random.choice(range(len(batch[0])), min(self._n_vis_images, len(batch[0])), replace=False)
            batch = [item[inds] for item in batch]
        
        image, semseg, wngy = batch
        y_hat = model.process(image)

        x = image.byte().cpu().numpy()

        y = semseg.permute(0, 3, 1, 2).mul(255).byte().cpu().numpy()
        pred_masks, pred_energies = y_hat
        pred_masks = pred_masks.mul(255).byte().unsqueeze(1).cpu().numpy()
        pred_energies = pred_energies.div(pred_energies.max()).mul(255).unsqueeze(1).byte().cpu().numpy()
        y_hat = np.concatenate((pred_masks, pred_energies), axis=1)

        images = []
        for x, true_masks, pred_masks in zip(x, y, y_hat):
            images.append(x)
            images.extend(true_masks)
            images.extend(pred_masks)

        self._visualizer.draw(images)
        grid = self._visualizer.get_image()
        if log_key is not None:
            model.logger.experiment.add_image(log_key, grid, self.current_epoch, dataformats='HWC')
        return grid

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: SegmentationModelInterface, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        if batch_idx % self._freq == 0:
            fmt_len = len(str(trainer.max_epochs * trainer.num_training_batches))
            save_fp = self._save_dir / 'train_step_{}.jpg'.format(str(trainer.global_step).zfill(fmt_len))
            pl_module.eval()
            img = self._build_vis_image(batch, pl_module)
            pl_module.train()
            Image.fromarray(img).save(save_fp, 'JPEG')

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: SegmentationModelInterface, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        fmt_len = len(str(trainer.max_epochs * trainer.num_training_batches))
        save_fp = self._save_dir / 'val_step_{}_batch_{:03d}.jpg'.format(str(trainer.global_step).zfill(fmt_len), batch_idx)
        img = self._build_vis_image(batch, pl_module)
        Image.fromarray(img).save(save_fp, 'JPEG')

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: SegmentationModelInterface, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        fmt_len = len(str(trainer.max_epochs * trainer.num_training_batches))
        save_fp = self._save_dir / 'test_step_{}_batch_{:03d}.jpg'.format(str(trainer.global_step).zfill(fmt_len), batch_idx)
        img = self._build_vis_image(batch, pl_module)
        Image.fromarray(img).save(save_fp, 'JPEG')

