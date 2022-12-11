from tqdm import tqdm
from functools import partial

import torch
import numpy as np

from src import models
from src.utils.config_reader import Config, object_from_dict
from src.datamodule import SegmentationDataModule
from src.postprocessing import watershed_cut, watershed_energy
from src.metrics import InstanceAveragePrecision


@torch.no_grad()
def eval_dwt(loader, model, postproc_fn=watershed_cut, level=None, **kwargs):
    aps = []
    metric = InstanceAveragePrecision()
    for img, semseg, wngy, gt_labels in tqdm(loader):
        pred_masks, pred_energies = model.process(img.cuda())
        pred_masks = pred_masks.mul(255).squeeze(1).byte().cpu().numpy()
        pred_energies = pred_energies.squeeze(1).byte().cpu().numpy()
        pred_labels = [postproc_fn(*args, **kwargs) for args in zip(pred_masks, pred_energies)]
        ap = metric(pred_labels, gt_labels.cpu().numpy(), level=level)
        aps.append(ap)
    return np.mean(aps)


if __name__ == '__main__':
    cfg = Config('configs/airs_pretrain_unet_dwt.yaml')

    data_module = SegmentationDataModule(cfg)
    data_module.prepare_data()

    data_module.train_dataset.compute_cc = data_module.val_dataset.compute_cc = data_module.test_dataset.compute_cc = True

    model = object_from_dict(cfg.model, models, config=cfg)
    model.load(drop_last=False)
    model.eval()
    model.cuda()

    postproc_fn = partial(watershed_cut, threshold=1)

    # result = eval_dwt(data_module.val_dataloader(), model, postproc_fn)
    result = eval_dwt(data_module.test_dataloader(), model, postproc_fn)
    print('AP:', result)
