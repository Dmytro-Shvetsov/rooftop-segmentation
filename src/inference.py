from pathlib import Path
from typing import Union
from collections import OrderedDict

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch_toolbelt.inference.tiles import ImageSlicer, TileMerger

from src import models
from src.datasets.inmem import InMemoryDataset
from src.utils.config_reader import Config, object_from_dict
from src.postprocessing import watershed_cut, watershed_energy, instance_to_rgb
from src.utils.io import read_image


class InferenceDriver:
    """Inference module for running a Deep Watershed Transform model on large satelite images in a tiled format."""
    def __init__(self, cfg:Union[str, Config]) -> None:
        '''
        Loads segmentation model and postprocessing modules. In addition, a logger is instantiated for inferencing videos.

        Args:
            cfg (Union[str, Config]): config object or path to a configuration file defining modules properties
        '''
        self.cfg = Config(cfg) if isinstance(cfg, str) else cfg
       
        self.device = torch.device(self.cfg.device)
        self.logs_dir = Path(self.cfg.logs_dir)
        self.logs_dir.mkdir(exist_ok=True, parents=True)

        self.segmentation:models.SegmentationModelInterface = object_from_dict(self.cfg.segmentation, parent=models, config=self.cfg)
        self.segmentation.eval()
        self.segmentation.load()
        self.segmentation.to(self.device)
        self._tile_size = self.cfg.tile_size # hxw
        self._tile_step = self.cfg.tile_step # hxw

    @torch.no_grad()
    def inference_image(self, image:Union[np.ndarray, torch.Tensor], **kwargs) -> Union[torch.Tensor, OrderedDict]:
        tile_slicer = ImageSlicer(image.shape, self._tile_size, self._tile_step, weight=self.cfg.tile_weight)
        tile_merger = TileMerger(tile_slicer.target_shape, 18, weight=tile_slicer.weight, device='cpu')
        patches = tile_slicer.split(image)

        data = list(
            {'image': patch, 'coords': np.array(coords, dtype=np.int)}
            for (patch, coords) in zip(patches, tile_slicer.crops)
        )
        for batch in tqdm(DataLoader(
            InMemoryDataset(data),
            pin_memory=True,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            drop_last=False,
        )):
            image = batch['image'].to(self.device)
            coords = batch['coords']
            output = self.segmentation.process(image, raw_outputs=True)
            tile_merger.integrate_batch(output.cpu(), coords)

        print()
        print('Merging the predictions...')
        pred_mask, pred_energy = self.segmentation.parse_outputs(tile_merger.merge().unsqueeze(0))

        pred_mask = tile_slicer.crop_to_orignal_size(pred_mask.squeeze().mul_(255).byte().cpu().numpy())
        pred_energy = tile_slicer.crop_to_orignal_size(pred_energy.squeeze().byte().cpu().numpy())

        print('Creating instance mask...')
        labels_mask = watershed_cut(
            pred_mask, 
            pred_energy,
            object_min_size=self.cfg.object_min_size,
            threshold=self.cfg.energy_threshold
        )
        return instance_to_rgb(labels_mask)

    def inference_images_directory(self, dir_path:Path, vis:bool=True) -> None:
        paths = list(filter(lambda x: x.suffix in {'.jpg', '.png', '.tiff', '.tif'}, dir_path.glob('**/*')))
        for path in tqdm(paths):
            print(f'Start processing {path.name}...')
            image = read_image(path, npy=True)
            image = cv2.resize(image, None, fx=self.cfg.target_scale, fy=self.cfg.target_scale, interpolation=cv2.INTER_CUBIC)
            
            labels_mask = self.inference_image(image)

            print('Saving results...')
            cv2.imwrite(str(self.logs_dir / path.name), cv2.cvtColor(labels_mask, cv2.COLOR_RGB2BGR))
            if vis:
                cv2.imwrite(str(self.logs_dir / f'vis_{path.name}'), cv2.cvtColor(cv2.addWeighted(image, 0.3, labels_mask, 0.7, 0.0), cv2.COLOR_RGB2BGR))
            print()
