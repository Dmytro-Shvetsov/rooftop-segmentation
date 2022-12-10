import cv2
import numpy as np
from pathlib import Path
from typing import Callable, Optional, List, Union
from multiprocessing.pool import Pool
from albumentations import pad_with_params
from logging import getLogger


logger = getLogger(__file__)

def _mp_initializer(shared_arr_):
    global shared_arr
    shared_arr = shared_arr_ # must be inherited, not passed as an argument


class TilingModule:
    def __init__(self, tile_size:int=(512, 512), stride:int=(384, 384), n_workers:int=1, format:str='.png', saver:Optional[Callable]=None):
        self.stride = stride # vertical x horisontal
        self.tile_size = tile_size # hxw
        self.n_workers = n_workers
        self.format = format
        self.saver = saver

    # @profile
    def tile_images(self, img:np.ndarray, mask:np.ndarray, save_paths:Optional[List[Path]]=None) -> List[Union[np.ndarray, str]]:
        assert img.shape[:2] == mask.shape[:2], 'All images should be of the same size.'

        if save_paths is not None:
            assert len(save_paths) == 2, 'Number of images should be the same as the number of saving paths.'
        h, w = img.shape[:2]
        xs, ys = np.meshgrid(np.arange(0, w, self.stride[1]), np.arange(0, h, self.stride[0]))

        map_args = [(xx, yy, save_paths) for xx, yy in zip(xs, ys)]

        if self.n_workers == 1:
            global shared_arr
            shared_arr = img, mask
            return [self._process_row(*args) for args in map_args]

        with Pool(self.n_workers, initializer=_mp_initializer, initargs=((img, mask),)) as workers_pool:
            return workers_pool.starmap(self._process_row, map_args)

    def _process_row(self, xs:List[int], ys:List[int], save_paths:Optional[List[Path]]=None) -> List[Union[np.ndarray, str]]:
        global shared_arr
        img, mask = shared_arr
        out_img_dir, out_mask_dir = save_paths if save_paths else (None, None)
 
        tileh, tilew = self.tile_size
        h, w, c = img.shape

        proc_result = []
        for x, y in zip(xs, ys):
            tile_img = img[y:y+tileh, x:x+tilew, :min(c, 3)]
            tile_mask = mask[y:y+tileh, x:x+tilew]

            img_invalid = tile_img.shape[0] < tileh / 4 or tile_img.shape[1] < tilew / 4 or (tile_img == 0).all() or (tile_img == 255).all()
            mask_invalid = (tile_mask[..., 0] == 0).all() or (tile_mask[..., 0] == 255).all() if len(tile_mask.shape) > 2 else (tile_mask == 0).all() or (tile_mask == 255).all()
            if img_invalid or mask_invalid:
                continue

            pad_args = 0, max(0, self.tile_size[0] - (h - y)), 0, max(0, self.tile_size[1] - (w - x)), cv2.BORDER_CONSTANT, 0
            tile_img = pad_with_params(tile_img, *pad_args)
            tile_mask = pad_with_params(tile_mask, *pad_args)

            tile_id = f'tile_y={y:05}_x={x:05}'
            if save_paths is not None and self.saver is None:
                img_save_fp = out_img_dir / f'{tile_id}{self.format}'
                cv2.imwrite(str(img_save_fp), cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR))
                proc_result.append(img_save_fp)

                if len(mask.shape) == 3 and mask.shape[-1] not in {1, 3}:
                    mask_save_fp = str(out_mask_dir / f'{tile_id}.npy')
                    np.save(mask_save_fp, tile_mask)
                else:
                    mask_save_fp = out_mask_dir / f'{tile_id}{self.format}'
                    cv2.imwrite(str(mask_save_fp), cv2.cvtColor(tile_mask, cv2.COLOR_RGB2BGR))
                proc_result.append(mask_save_fp)
            else:
                proc_result.append((str(out_img_dir / tile_id) if out_img_dir else tile_id, tile_img))
                proc_result.append((str(out_mask_dir / tile_id) if out_mask_dir else tile_id, tile_mask))
        if self.saver is not None and any(proc_result):
            self.saver(proc_result)
        return proc_result


if __name__ == '__main__':
    h, w = 16, 16
    s = 4, 4
    ts = 4, 4

    img = np.random.randn(h, w)

    tm = TilingModule(ts, s)
    res = tm.tile_images([img])
    print(len(res), len(res[0]), len(res[0][0]))
    print(res[0][0])
