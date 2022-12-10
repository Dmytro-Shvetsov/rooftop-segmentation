import sys
import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from src.datasets.tiling import TilingModule
from src.utils.eval import eval_expr
from src.utils.io import read_image
from src.models.dwt import semg_to_dtfm, dtfm_to_wngy

argparser = argparse.ArgumentParser(description='Script for tiling a set of large aerial images to build a dataset.')
argparser.add_argument('-id', '--images_dir', type=str, required=True, help='Directory containing tiff images to be tiled.')
argparser.add_argument('-ad', '--annots_dir', type=str, required=True, help='Directory containing ground truth segmentation masks (binary or multiclass).')
argparser.add_argument('-od', '--output_dir', type=str, required=True, help='Output directory where to write the resulting dataset.')
argparser.add_argument('-ts', '--tile_size', type=int, nargs='+', default=(512, 512), help='Height x Width of a single tile.')
argparser.add_argument('-st', '--stride', type=int, nargs='+', default=(384, 384), help='Vertical x Horisontal step size between tiles.')
argparser.add_argument('-sc', '--scale', type=str, default='1.0', help='Scaling factor for images and their masks.')
argparser.add_argument('-fmt', '--format', type=str, default='.jpg', choices=('.png', '.jpg'), help='Target dataset\'s format to be created.')
argparser.add_argument('-w', '--workers', type=int, default=8, help='Number of workers to spawn for tiling.')
argparser.add_argument('-dwt', '--deep_watershed_transform', action='store_true', default=False, help='Whether to precompute the watershed energies for the masks for the DWT baseline.')


def main(args):
    images_dir, annots_dir, output_dir = Path(args.images_dir), Path(args.annots_dir), Path(args.output_dir)
    assert images_dir.exists() and images_dir.is_dir(), f'Invalid images directory path {images_dir}.'
    assert annots_dir.exists() and annots_dir.is_dir(), f'Invalid annotations directory path {annots_dir}.'

    image_paths = list(images_dir.glob('*.tif'))
    image_paths.sort()
    
    output_dir.mkdir(exist_ok=True, parents=True)

    tm = TilingModule(args.tile_size, args.stride, args.workers, format=args.format, saver=None)
    start_t = perf_counter()

    scale = eval_expr(args.scale)

    assert bool(image_paths), 'No .tif images found'

    for img_path in image_paths:
        img_name = img_path.stem
        ann_path = next(annots_dir.glob(f'{img_name}.*'), None)

        if not ann_path:
            raise RuntimeError(f'Unable to find ground truth mask for image {img_path} in {annots_dir}.')

        img_tiles_dir = output_dir / 'images' / img_name
        mask_tiles_dir = output_dir / 'masks' / img_name

        img_tiles_dir.mkdir(exist_ok=True, parents=True)
        mask_tiles_dir.mkdir(exist_ok=True, parents=True)

        img, mask = read_image(img_path, npy=True), read_image(ann_path, npy=True)
        if scale != 1.0:
            img = cv2.resize(img, None, fy=scale, fx=scale, interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(mask, None, fy=scale, fx=scale, interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST)

        if mask.max() == 1:
            mask *= 255

        if args.deep_watershed_transform:
            if len(mask.shape) == 3: 
                binary = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                binary = (binary > 0).astype(np.uint8)
            else:
                binary = mask
            mask = np.concatenate([mask[..., np.newaxis], dtfm_to_wngy(semg_to_dtfm(binary))], axis=-1).astype(np.uint8)

        print(f'Tiling image {img_name} and its mask...')
        result = tm.tile_images(img, mask, (img_tiles_dir, mask_tiles_dir))
        num_created = sum(len(r) // 2 for r in result)
        if num_created == 0:
            img_tiles_dir.rmdir()
            mask_tiles_dir.rmdir()
        print(f'Сreated {num_created} dataset examples.')
    print(f'Successfully processed all of the annotations. Scale: {args.scale}. Time: {perf_counter() - start_t}')

    with open(output_dir / 'command.txt', 'a') as fid:
        fid.write('python ' + ' '.join(sys.argv) + '\n')


if __name__ == '__main__':
    main(argparser.parse_args())
