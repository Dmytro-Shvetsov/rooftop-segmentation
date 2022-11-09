from pathlib import Path
import cv2
import json
from datetime import date
import argparse

def get_contours(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def get_boundings(img):
    contours = get_contours(img)
    bounding_boxes = []
    for i in range(len(contours)):
        bounding_box = cv2.boundingRect(contours[i])
        bounding_boxes.append(bounding_box)

    return bounding_boxes

def get_size(img):
    return img.shape[:2]

def flatten_contour_coordinates(contours):
    flt = list(contours.reshape(1, 2* len(contours)).flatten())
    flt = [int(x) for x in flt]
    return flt

def images_to_dict(imgs_path):
    IMAGES = dict()
    IMAGES['images'] = []

    for i in range(len(imgs_path)):
        file = dict()
        img = cv2.imread(str(imgs_path[i]))
        h, w = get_size(img)

        file['filename'] = str(imgs_path[i])
        file['width'] = h
        file['height'] = w
        file['id'] = i
        
        IMAGES['images'].append(file)

    return IMAGES

def annotations_to_dict(msks_path):
    ANNOTATIONS = dict()
    ANNOTATIONS['annotations'] = []

    counter = 0
    for j in range(len(msks_path)):
        mask = cv2.imread(str(msks_path[j]))
        contours = get_contours(mask)
        bbox = get_boundings(mask)
        
        for i in range(len(contours)):
            stats = dict()
            rooftop_contour = contours[i]
            rooftop_bbox = list(bbox[i])
            stats['segmentation'] = []
            stats['segmentation'].append(flatten_contour_coordinates(rooftop_contour))
            stats['area'] = rooftop_bbox[2] * rooftop_bbox[3]
            stats['is_crowd'] = 0
            stats['image_id'] = j
            stats['bbox'] = rooftop_bbox
            stats['category_id'] = 0
            stats['id'] = counter
            
            counter += (i + 1)
            ANNOTATIONS['annotations'].append(stats)

    return ANNOTATIONS

def categories_to_dict():
    CATEGORIES = dict()
    CATEGORIES['categories'] = [{'id': 0, 'name': 'rooftop'}]
    return CATEGORIES
    
def info_to_dict():
    today = date.today().strftime("%d/%m/%Y")
    INFO = dict()
    INFO['info'] = {
                    "description": "Rooftop Segmentation Dataset",
                    "url": "https://github.com/Dmytro-Shvetsov/rooftop-segmentation",
                    "version": "1.0",
                    "year": 2022,
                    "contributor": "GOLOVACHI",
                    "date_created": str(today)
                   }
    
    return INFO

def generate_COCO(args):
    path_to_AIRS, outputfile = Path(args.input_dir), Path(args.output)

    assert path_to_AIRS.exists() and path_to_AIRS.is_dir(), f'Invalid directory path {path_to_AIRS}.'
    assert outputfile.exists(), f'No JSON file provided with path {outputfile}'

    path_to_masks = path_to_AIRS / 'masks'
    path_to_images = path_to_AIRS / 'images'

    images_pathes = list(path_to_images.glob('**/*.png'))
    masks_pathes = list(path_to_masks.glob('**/*.png'))

    images = images_to_dict(images_pathes)
    annotations = annotations_to_dict(masks_pathes)
    categories = categories_to_dict()
    info = info_to_dict()

    COCO = dict()
    COCO['info'] = info['info']
    COCO['images'] = images['images']
    COCO['annotations'] = annotations['annotations']
    COCO['categories'] = categories['categories']

    
    file = open(outputfile, 'w')
    json.dump(COCO, file, indent= 4)
    file.close()


# path = Path("./AIRS_binary1")

# generate_COCO(path, 'COCO.json')


argparser = argparse.ArgumentParser(description='Script for creating a COCO annotations.')
argparser.add_argument('-id', '--input_dir', type=str, required=True, help='Directory containing dataset.')
argparser.add_argument('-od', '--output', type=json, required=True, help='JSON file to write to')

if __name__ == '__main__':
    generate_COCO(argparser.parse_args())






