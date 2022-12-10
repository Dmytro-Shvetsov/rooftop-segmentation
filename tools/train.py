import os
import argparse
from json import dump as json_dump

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src import models
from src.datamodule import SegmentationDataModule
from src import callbacks as cb
from src.utils.config_reader import Config, object_from_dict


argparser = argparse.ArgumentParser(description='Python script used for model training using given configuration file.')
argparser.add_argument('-c', '--config', type=str, required=True, dest='cfg_path', help='Configuration file path.')
argparser.add_argument('-tlr', '--tune_lr', required=False, default=False, action='store_true', help='Automatically estimate the best learning rate.')


def dump_test_results(outs, file_path, **kwargs):
    outs = {k: round(v, 4) for k, v in outs.items()}
    outs.update(kwargs)
    with open(file_path, 'w', encoding='utf8') as fid:
        json_dump(outs, fid, indent=4)


def train(args):
    cfg = Config(args.cfg_path)
    seed_everything(cfg.seed)

    data_module = SegmentationDataModule(cfg)
    model:LightningModule = object_from_dict(cfg.model, parent=models, config=cfg)
    model.load(drop_last=cfg.get('drop_last', True))

    logger = TensorBoardLogger(cfg.logs_dir, cfg.run_name, default_hp_metric=False, version=cfg.experiment_version)

    callbacks = [
        ModelCheckpoint(**cfg.checkpoint_params),
        LearningRateMonitor(logging_interval='step'),
        cb.DWTVisualizationCallback(cfg, save_dir=os.path.join(logger.log_dir, 'visualizations'), **cfg.vis_settings),
    ]

    trainer:Trainer = object_from_dict(cfg.trainer, callbacks=callbacks, logger=logger)
    if args.tune_lr:
        print('Finding optimal learning rate...')
        lr_finder = trainer.tuner.lr_find(model, data_module)
        lr = lr_finder.suggestion()
        cfg.optimizer['lr'] = lr
        print(f'Using learning rate for training: {lr}.')

    trainer.fit(model, data_module)

    cfg.model_ckpt = trainer.checkpoint_callback.best_model_path
    model.load()
    results = trainer.test(model, data_module)[0]
    dump_test_results(results, os.path.join(logger.log_dir, 'test_results.json'), model_ckpt=cfg.model_ckpt)
    cfg.copy(os.path.join(logger.log_dir, 'hparams.yaml'))
    print(f'Finished training run and saved results in {repr(logger.log_dir)}')


if __name__ == '__main__':
    train(argparser.parse_args())
