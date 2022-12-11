# **Setting up the repository:**
After cloning the repository and installing the (Nvidia) Docker:
```
$ docker build -f ./docker/DockerfileCUDA11 --rm -t rooftop .

$ docker run --ipc=host --name rooftopml --gpus all -it --network host -v "$PWD:/home/docker/repository" rooftop /bin/bash
```
If you are using a machine without Nvidia GPU, remove the `--gpus all` parameter.

### Enter container to run some experiments.
```
docker start -i rooftopml
```

# **Image tiling:**

```
python tools/create_images_dataset.py -id {path to images} -ad {path to masks} -od {output directory} -ts 512 512 -st 512 512 -sc 0.3 -fmt .png -w 12
```

**Arguments:**

- `-ts` size of tiles (x, y)
- `-st` stride (x, y)
- `-sc` Scale factor
- `-fmt` output image format
- `-w` number of workers to be spawned for tiling

# **Building models on AIRS dataset**

## Deep Watershed Transform (DWT) segmentation model:
```
python tools/train.py -c configs/airs_pretrain_unet_dwt.yaml
```

#### To be able to use mmdetection library, please refer to the original documentation of the library to [install](https://mmdetection.readthedocs.io/en/stable/get_started.html) the package inside the container.

## Generating COCO annotations (train/val/test)
```
python tools/coco_annotations.py -i data/airs_proto/train -o data/airs_proto/train_annotations.json --minimize
```
```
python tools/coco_annotations.py -i data/airs_proto/val -o data/airs_proto/val_annotations.json --minimize
```
```
python tools/coco_annotations.py -i data/airs_proto/test -o data/airs_proto/test_annotations.json --minimize
```

Arguments:

- `-id` Directory containing dataset
- -`od` JSON file to write to
- `-m` Whether to minimize the output json file or not

## Training [Point Rend](https://github.com/open-mmlab/mmdetection/tree/master/configs/point_rend)
```
python mmdetection/tools/train.py configs/pointrend_r50.py
```

# **Inferencing a folder of GeoTiff images with DWT model**
```
python tools/inference_geotiff.py -c configs/airs_pretrain_unet_dwt_inference.yaml -id data/inference_examples/
```
