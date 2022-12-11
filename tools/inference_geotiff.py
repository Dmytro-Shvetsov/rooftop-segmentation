import argparse
from pathlib import Path

from src.inference import InferenceDriver

argparser = argparse.ArgumentParser(description='Python script used for inferencing a large GeoTiff image from model using given configuration file.')
argparser.add_argument('-c', '--config', type=str, required=True, dest='cfg_path', help='Configuration file path.')
argparser.add_argument('-id', '--images_dir', type=str, required=True, help='A directory containing images to be inferenced.')

def main(args):
    inf = InferenceDriver(args.cfg_path)
    img_dir = Path(args.images_dir)
    assert img_dir.exists() and img_dir.is_dir(), 'Invalid images directory path.'

    inf.inference_images_directory(img_dir)
    print(f'Finish inference run and saved results in \'{inf.logs_dir}\'')


if __name__ == '__main__':
    main(argparser.parse_args())
