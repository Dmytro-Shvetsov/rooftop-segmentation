from os import PathLike
from typing import Union
from pathlib import Path

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

try:
    import tiffile as tiff
    read_tiff = tiff.imread
except ImportError:
    print('tiffile package not installed, using pillow')
    read_tiff = Image.open

def read_image(path:PathLike, npy:bool=False) -> Union[np.ndarray, Image.Image]:
    path = Path(path)
    assert path.is_file(), f'{path} is not a file.'
    img = read_tiff(path) if 'tif' in path.suffix else Image.open(path)
    return np.asarray(img) if npy else img

