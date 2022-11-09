import sys
import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from src.datasets.tiling import TilingModule
from src.utils.eval import eval_expr
from src.utils.io import read_image

argparser = argparse.ArgumentParser(description='Script for tiling a set of large aerial images to build a dataset.')
argparser.add_argument('-id', '--images_dir', type=str, required=True, help='Directory containing tiff images to be tiled.')
argparser.add_argument('-ad', '--annots_dir', type=str, required=True, help='Directory containing ground truth segmentation masks (binary or multiclass).')
argparser.add_argument('-od', '--output_dir', type=str, required=True, help='Output directory where to write the resulting dataset.')
argparser.add_argument('-ts', '--tile_size', type=int, nargs='+', default=(512, 512), help='Height x Width of a single tile.')
argparser.add_argument('-st', '--stride', type=int, nargs='+', default=(384, 384), help='Vertical x Horisontal step size between tiles.')
argparser.add_argument('-sc', '--scale', type=str, default='1.0', help='Scaling factor for images and their masks.')
argparser.add_argument('-fmt', '--format', type=str, default='.jpg', choices=('lmdb', '.png'), help='Target dataset\'s format to be created.')
argparser.add_argument('-w', '--workers', type=int, default=8, help='Number of workers to spawn for tiling.')



def calculate_dataset_size(num_images, img_size, tile_size, stride, num_channels=4):
    tile_dims = (int(np.ceil((img_size[0] - tile_size[0]) / stride[0]) + 1), # vertical
                 int(np.ceil((img_size[1] - tile_size[1]) / stride[1]) + 1))  # horizontal
    channel_mem_size = np.int64(np.zeros(tile_size, dtype=np.uint8).nbytes * 10)
    # print( np.prod(tile_dims), tile_dims)
    # print(channel_mem_size, num_images,)
    return num_images * np.prod(tile_dims) * (num_channels * channel_mem_size)


def main(args):
    images_dir, annots_dir, output_dir = Path(args.images_dir), Path(args.annots_dir), Path(args.output_dir)
    assert images_dir.exists() and images_dir.is_dir(), f'Invalid images directory path {images_dir}.'
    assert annots_dir.exists() and annots_dir.is_dir(), f'Invalid annotations directory path {annots_dir}.'

    image_paths = list(images_dir.glob('*.tif'))
    image_paths.sort()
    
    output_dir.mkdir(exist_ok=True, parents=True)

    img_size = read_image(image_paths[0]).shape[:2]
    size = calculate_dataset_size(len(image_paths), img_size, args.tile_size, args.stride, 4)
    # print(size)
    # exit()
    # saver = LMDB(output_dir, memsize=1e+9) if args.format.lower() == 'lmdb' else None
    saver = None

    tm = TilingModule(args.tile_size, args.stride, args.workers, format=args.format, saver=saver)
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

        if args.format != 'lmdb':
            img_tiles_dir.mkdir(exist_ok=True, parents=True)
            mask_tiles_dir.mkdir(exist_ok=True, parents=True)

        img, mask = read_image(img_path, npy=True), read_image(ann_path, npy=True)
        if scale != 1.0:
            img = cv2.resize(img, None, fy=scale, fx=scale, interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(mask, None, fy=scale, fx=scale, interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST)

        if mask.max() == 1:
            mask *= 255

        print(f'Tiling image {img_name} and its mask...')
        result = tm.tile_images(img, mask, (img_tiles_dir, mask_tiles_dir))
        num_created = sum(len(r) // 2 for r in result)
        if args.format != 'lmdb' and num_created == 0:
            img_tiles_dir.rmdir()
            mask_tiles_dir.rmdir()
        print(f'Сreated {num_created} dataset examples.')
    print(f'Successfully processed all of the annotations. Scale: {args.scale}. Time: {perf_counter() - start_t}')

    with open(output_dir / 'command.txt', 'a') as fid:
        fid.write('python ' + ' '.join(sys.argv) + '\n')


if __name__ == '__main__':
    main(argparser.parse_args())
