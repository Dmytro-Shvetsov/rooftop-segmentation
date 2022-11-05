# **Image tiling:**

python tools/create_images_dataset.py -id {path to images} -ad {path to masks} -od {output directory} -ts 512 512 -st 512 512 -sc 0.3 -fmt .png -w 12

**Arguments:**

- -ts size of tiles (x, y)
- -st stride (x, y)
- -sc Scale factor
- -fmt output image format
- -w number of workers to be spawned for tiling


