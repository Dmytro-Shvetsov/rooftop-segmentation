import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    assert len(sys.argv) >= 2, 'No images directory provided. Example usage: `python tools/compute_norm_params.py ./dataset/train/images`'

    images_dir = Path(sys.argv[1])
    assert images_dir.exists(), 'Invalid directory path.'

    mean_sum = np.array([0.0, 0.0, 0.0])
    std_sum = np.array([0.0, 0.0, 0.0])
    count = 0

    for fp in tqdm(list(images_dir.glob('**/*.png'))):
        img = np.array(Image.open(fp)) / 255
        mean_sum += img.mean(axis=(0, 1))
        std_sum += img.std(axis=(0, 1))
        count += 1

    print('Mean:', str(mean_sum / count * 255))
    print('Std:', str(std_sum / count * 255))


if __name__ == '__main__':
    main()
