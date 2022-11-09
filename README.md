# **Image tiling:**

python tools/create_images_dataset.py -id {path to images} -ad {path to masks} -od {output directory} -ts 512 512 -st 512 512 -sc 0.3 -fmt .png -w 12

**Arguments:**

- -ts size of tiles (x, y)
- -st stride (x, y)
- -sc Scale factor
- -fmt output image format
- -w number of workers to be spawned for tiling

# **COCO annotations** 

python tools/coco_annotations.py -i  airs_proto/train -o airs_proto/train_annotations.json --minimize <br>
python tools/coco_annotations.py -i  airs_proto/val -o airs_proto/val_annotations.json --minimize <br>
python tools/coco_annotations.py -i  airs_proto/test -o airs_proto/test_annotations.json --minimize <br>

**Arguments:**

- -id Directory containing dataset
- -od JSON file to write to
- -m Whether to minimize the output json file or not


