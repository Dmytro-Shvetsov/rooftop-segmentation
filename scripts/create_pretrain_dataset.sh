python tools/create_images_dataset.py -id ./data/AIRS/train/image/ -ad data/AIRS/train/label/ -od ./data/airs_proto_dwt/train -ts 512 512 -st 512 512 -sc 0.3 -fmt .png -w 12 -dwt

python tools/create_images_dataset.py -id ./data/AIRS/val/image/ -ad data/AIRS/val/label/ -od ./data/airs_proto_dwt/val -ts 512 512 -st 512 512 -sc 0.3 -fmt .png -w 12 -dwt

python tools/create_images_dataset.py -id ./data/AIRS/test/image/ -ad data/AIRS/test/label/ -od ./data/airs_proto_dwt/test -ts 512 512 -st 512 512 -sc 0.3 -fmt .png -w 12 -dwt
