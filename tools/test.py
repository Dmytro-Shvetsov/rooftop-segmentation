import os
import argparse
from json import dump as json_dump

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from src import models
from src import callbacks as cb
from src.datamodule import SegmentationDataModule
from src.utils.config_reader import Config, object_from_dict


argparser = argparse.ArgumentParser(description='Python script used for model evaluation using given configuration file.')
argparser.add_argument('-c', '--config', type=str, required=True, dest='cfg_path', help='Configuration file path.')


def dump_test_results(outs, file_path, **kwargs):
    outs = {k: round(v, 4) for k, v in outs.items()}
    outs.update(kwargs)
    with open(file_path, 'w', encoding='utf8') as fid:
        json_dump(outs, fid, indent=4)


def test(args):
    cfg = Config(args.cfg_path)
    seed_everything(cfg.seed)

    data_module = SegmentationDataModule(cfg)
    model:LightningModule = object_from_dict(cfg.model, parent=models, config=cfg)
    model.load()

    logger = TensorBoardLogger(cfg.logs_dir, cfg.run_name, default_hp_metric=False, version=cfg.experiment_version)
    callbacks = [
        cb.DWTVisualizationCallback(cfg, save_dir=os.path.join(logger.log_dir, 'visualizations'), **cfg.vis_settings),
    ]
    trainer:Trainer = object_from_dict(cfg.trainer, logger=logger, callbacks=callbacks)

    results = trainer.test(model, data_module)[0]
    dump_test_results(results, os.path.join(logger.log_dir, 'test_results.json'), model_ckpt=cfg.model_ckpt)
    cfg.copy(os.path.join(logger.log_dir, 'hparams.yaml'))
    print(f'Saved validation results in {repr(logger.log_dir)}')


if __name__ == '__main__':
    test(argparser.parse_args())
