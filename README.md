# pytorch-yolov3

![](output/herd_of_horses.png)

## About

simple Pytorch implementation of YOLOv3 / YOLOv3-tiny.

## Features

- [x]: Support darknet format weights
- [x]: Sample code for infering image and video

## Usage

### Download weights

First of all, download the official weights.

```bash
./weights/download_weights.sh
```

### Infer a single image (YOLOv3)

```bash
python detect_image.py \
    --input data/dog.png \
    --output output \
    --weights weights/yolov3.weights \
    --config config/yolov3_coco.yaml
```

### Infer images in the dicretory.

```bash
python detect_image.py \
    --input data \
    --output output \
    --weights weights/yolov3.weights \
    --config config/yolov3_coco.yaml
```

### Infer video

```bash
python detect_video.py \
    --input data/sample.avi \
    --output output \
    --weights weights/yolov3.weights \
    --config config/yolov3_coco.yaml
```
