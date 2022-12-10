# Deep Watershed Transform for Instance Segmentation
# https://arxiv.org/abs/1611.08303
# 
# The implementation is inspired by this notebook:
# https://www.kaggle.com/code/ebinan92/unet-with-deep-watershed-transform-dwt-train

from typing import Union
import cv2
import torch
import numpy as np
import pytorch_lightning as pl

from torchvision.transforms import Compose, Normalize
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.losses import FocalLoss

from src.utils.config_reader import object_from_dict
from src.models.base import SegmentationModelInterface


WATERSHED_ENERGY_BINS = np.array(
    [0.955,  1.369,  1.91,  2.324,  2.739,  3.279,  3.694,  4.108,
     4.649,  5.063,  5.477,  6.018, 10., 20., 40., 80.])


def dtfm_to_wngy(dtfm):
    '''
    Distance transform to watershed energy.

    Args:
        dtfm (np.array, np.float32): Distance transform.
            Shape (H, W, 1). 
    Returns:
        wngy (np.array, np.int64): Watershed energy levels map.
            Shape (H, W, 1).
    '''
    wngy = np.digitize(dtfm, bins=WATERSHED_ENERGY_BINS)

    return wngy


def semg_to_dtfm(semg):
    '''
    Convert semantic segmentation to distance transform.

    Args:
        semg (np.array, np.float32): Semantic segmentation.
            Shape (H, W, 1). Values 0, or 1.
    
    Returns:
        dtfm (np.array, np.float32): Distance transform.
            Shape (H, W, 1). 
    '''
    dtfm = cv2.distanceTransform(semg.astype(np.uint8),
                                 distanceType=cv2.DIST_L2,
                                 maskSize=3)
    dtfm = dtfm[..., None]

    return dtfm


class WatershedEnergyLitModel(pl.LightningModule, SegmentationModelInterface):
    _is_loaded = False

    def __init__(self, config, **kwargs):
        super().__init__()
        self.wsgy_model = Unet('resnet50', in_channels=3, classes=18, encoder_weights=None)
        self._cfg = config
        self._ckpt_path =  kwargs.get('model_ckpt', self._cfg.get('model_ckpt'))

        self.lr = self._cfg.get('lr', 5e-4)
        self.one_cycle_max_lr = self._cfg.get('one_cycle_max_lr', None)
        self.one_cycle_total_steps = self._cfg.get('one_cycle_total_steps', 25)
        self.optimizer = self._cfg['optimizer']

        self.preprocess = Compose([
            Normalize(self._cfg.means, self._cfg.stds, inplace=True),
        ])
        
        self._metrics = {item['type'].split('.')[-1]: object_from_dict(item).to(self._device) for item in config.get('metrics', [])}
        self.classes = ('Roof',)

        self._semseg_losses = {item['type'].split('.')[-1]: (item.pop('weight', 1.0), object_from_dict(item, mode='binary')) for item in config.get('losses', [], copy=True)}

        self._wsgy_losses = {'FocalLoss': (1.0, FocalLoss('multiclass', gamma=2))}

    def forward(self, x):
        return self.wsgy_model(x)

    def load(self, drop_last=False) -> bool:
        if not self._ckpt_path:
            print('Unable to find the model checkpoint parameter. The model was not loaded')
            return self._is_loaded

        if not self.is_loaded:
            ckpt = torch.load(self._ckpt_path, map_location=self._device)
            if drop_last:
                for k in list(ckpt['state_dict'].keys())[-2:]:
                    ckpt['state_dict'].pop(k)
            if 'engine' in ckpt:
                # import trt module here as it may not be installed for cpu-only environments
                from torch2trt import TRTModule
                self.wsgy_model = TRTModule()
                self.wsgy_model.load_state_dict(ckpt)
            else:
                self.wsgy_model.load_state_dict({k.replace('wsgy_model.', ''):v for k, v in ckpt['state_dict'].items()}, strict=not drop_last)
            self._is_loaded = True
            print(f'Loaded segmentation model from {repr(self._ckpt_path)}')
        self.wsgy_model.to(self._device)
        return self._is_loaded

    def preprocess_inputs(self, images: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).to(self._device)
        if images.size(1) not in {1, 3}:
            images = images.permute(0, 3, 1, 2)

        images = self.preprocess(images.float().div_(255))
        return images

    def parse_outputs(self, outs):
        return (outs[:, 0] > 0).byte(), outs[:, 1:].argmax(dim=1)

    def configure_optimizers(self):
        optimizer = object_from_dict(self._cfg.optimizer, params=self.parameters())
        scheduler = object_from_dict(self._cfg.scheduler, optimizer=optimizer)
        scheduler = {
            'scheduler': scheduler,
            'reduce_on_plateau': True,
            'monitor': self._cfg.checkpoint_params['monitor']
        }
        return [optimizer], [scheduler]

    def _step(self, batch, log_prefix, metrics=False):
        img, semg, wngy = batch

        # uvec = uvec.permute(0, 3, 1, 2)
        wngy = wngy.permute(0, 3, 1, 2).long().squeeze()
        semg = semg.permute(0, 3, 1, 2)
        # area = area.permute(0, 3, 1, 2)

        img = self.preprocess_inputs(img)
        y_hat = self(img)

        total_loss = 0.0

        # semantic segmentation loss
        for loss_name, (weight, loss) in self._semseg_losses.items():
            err = weight * loss(y_hat[:, 0].unsqueeze(1), semg.squeeze(1))
            total_loss += err
            self.log(f'{log_prefix}/semseg_losses/{loss_name}', err, logger=True, on_step=False, on_epoch=True)
            # self.log('train_semseg_'+loss_name, err, logger=True, on_step=False, on_epoch=True)

        # watershed energies loss
        for loss_name, (weight, loss) in self._wsgy_losses.items():
            err = weight * loss(y_hat[:, 1:], wngy)
            total_loss += err
            self.log(f'{log_prefix}/wngy_losses/{loss_name}', err, logger=True, on_step=False, on_epoch=True)

        # loss = self.train_loss(y_hat, wngy, semg, area)
        # self.log('train_loss', total_loss, on_step=False, on_epoch=True)
        self.log(f'{log_prefix}/losses/total_loss', total_loss, on_step=True, on_epoch=True)

        if metrics:
            y_hat = self.parse_outputs(y_hat)[0].unsqueeze(1)
            for metric_name, metric in self._metrics.items():
                avg_score = 0.0
                for i, c in enumerate(self.classes):
                    score = metric(y_hat[:, i], semg[:, i])
                    avg_score += score / (len(self.classes) - 1)
                    self.log(f'{log_prefix}/metrics/{metric_name}_{c.lower()}', score, on_epoch=True)
                self.log(f'{log_prefix}/metrics/{metric_name}', avg_score, on_epoch=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'Train', True)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'Validation', True)

    def test_step(self, batch, batch_idx):
        return self._step(batch, 'Test', True)
