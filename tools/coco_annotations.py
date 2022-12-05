import argparse
import json
from pathlib import Path
from datetime import date

import cv2

argparser = argparse.ArgumentParser(description='Script for creating a COCO annotations.')
argparser.add_argument('-id', '--input_dir', type=str, required=True, help='Directory containing dataset.')
argparser.add_argument('-od', '--output', type=str, required=True, help='JSON file to write to')
argparser.add_argument('-m', '--minimize', action='store_true', default=False, help='Whether to minimize the output json file or not')


def get_contours(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_boundings(contours):
    bounding_boxes = list(map(cv2.boundingRect, contours))
    return bounding_boxes


def get_size(img):
    return img.shape[:2]


def flatten_contour_coordinates(contours):
    return contours.reshape(-1).tolist()


def images_to_dict(imgs_path):
    IMAGES = []
    for i in range(len(imgs_path)):
        img = cv2.imread(str(imgs_path[i]))
        h, w = get_size(img)
        file = {
            'id': i,
            'file_name': imgs_path[i].parts[-2] + '/' + imgs_path[i].parts[-1],
            'width': w,
            'height': h,
        }
        IMAGES.append(file)
    return IMAGES


def annotations_to_dict(msks_path):
    ANNOTATIONS = []
    counter = 0
    for j in range(len(msks_path)):
        mask = cv2.imread(str(msks_path[j]))
        contours = get_contours(mask)
        bbox = get_boundings(contours)
        for i in range(len(contours)):
            # https://github.com/cocodataset/cocoapi/issues/139
            if len(contours[i]) <= 2:
                continue
            stats = {
                'id': counter,
                'segmentation': [flatten_contour_coordinates(contours[i])],
                'area': cv2.contourArea(contours[i]),
                'iscrowd': 0,
                'image_id': j,
                'bbox': bbox[i],
                'category_id': 0,
            }
            counter += (i + 1)
            ANNOTATIONS.append(stats)
    return ANNOTATIONS


def categories_to_dict():
    return [{'id': 0, 'name': 'rooftop'}]


def info_to_dict():
    today = date.today().strftime("%d/%m/%Y")
    return {
        "description": "Rooftop Segmentation Dataset",
        "url": "https://github.com/Dmytro-Shvetsov/rooftop-segmentation",
        "version": "1.0",
        "year": 2022,
        "contributor": "GOLOVACHI",
        "date_created": str(today)
    }


def generate_COCO(args):
    path_to_AIRS, outputfile = Path(args.input_dir), Path(args.output)

    assert path_to_AIRS.exists() and path_to_AIRS.is_dir(), f'Invalid directory path {path_to_AIRS}.'
    assert outputfile.parent.exists(), f'No JSON file provided with path {outputfile}'

    path_to_masks = path_to_AIRS / 'masks'
    path_to_images = path_to_AIRS / 'images'

    images_pathes = list(path_to_images.glob('**/*.png'))
    masks_pathes = list(path_to_masks.glob('**/*.png'))

    COCO = {
        'info': info_to_dict(),
        'images': images_to_dict(images_pathes),
        'annotations': annotations_to_dict(masks_pathes),
        'categories': categories_to_dict()
    }
    
    with open(outputfile, 'w') as file:
        if args.minimize:
            file.write(json.dumps(COCO))
        else:
            json.dump(COCO, file, indent=4)


if __name__ == '__main__':
    generate_COCO(argparser.parse_args())
