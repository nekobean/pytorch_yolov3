# pytorch-yolov3

![](output/herd_of_horses.png)

## About

simple Pytorch implementation of YOLOv3 / YOLOv3-tiny.

## Features

- [x]: Support darknet format weights
- [x]: Sample code for infering image and video
- [x]: Sample code for training

### Setup

1. Install dependent libraries.

    ```bash
    pip install -r requirements.txt
    ```

    If you have already installed older version of Pytorch, update it.

    ```bash
    pip install -U torch torchvision torchaudio
    ```

2. Download the official weights.

    ```bash
    ./weights/download_weights.sh
    ```

## Usage (YOLOv3)

### Detect from a single image

```bash
python detect_image.py \
    --input data/dog.png \
    --output output \
    --weights weights/yolov3.weights \
    --config config/yolov3_coco.yaml
```

### Detect from images in the dicretory.

```bash
python detect_image.py \
    --input data \
    --output output \
    --weights weights/yolov3.weights \
    --config config/yolov3_coco.yaml
```

### Detect from a single video

```bash
python detect_video.py \
    --input data/sample.avi \
    --output output \
    --weights weights/yolov3.weights \
    --config config/yolov3_coco.yaml
```

### Train custom dataset

[YOLOv3 - 自作データセットで学習する方法について - pystyle](https://pystyle.info/pytorch-yolov3-how-to-train-custom-dataset/)

```bash
python train_custom.py \
    --dataset_dir custom_dataset \
    --weights weights/darknet53.conv.74 \
    --config config/yolov3_custom.yaml
```

### Evaluate custom dataset

```bash
python evaluate_custom.py \
    --dataset_dir custom_dataset \
    --weights train_output/yolov3_final.pth \
    --config config/yolov3_custom.yaml
```

## Usage (YOLOv3-tiny)

### Detect from a single image

```bash
python detect_image.py \
    --input data/dog.png \
    --output output \
    --weights weights/yolov3-tiny.weights \
    --config config/yolov3tiny_coco.yaml
```

### Detect from images in the dicretory.

```bash
python detect_image.py \
    --input data \
    --output output \
    --weights weights/yolov3-tiny.weights \
    --config config/yolov3tiny_coco.yaml
```

### Detect from a single video

```bash
python detect_video.py \
    --input data/sample.avi \
    --output output \
    --weights weights/yolov3-tiny.weights \
    --config config/yolov3tiny_coco.yaml
```

### Train custom dataset

Download [YOLOv3-tiny pretrained weights](https://drive.google.com/file/d/1sEjehR9psSD9lWHvXABaN6jbAAkbwbBX/view?usp=sharing).

[YOLOv3 - 自作データセットで学習する方法について - pystyle](https://pystyle.info/pytorch-yolov3-how-to-train-custom-dataset/)

```bash
python train_custom.py \
    --dataset_dir custom_dataset \
    --weights weights/yolov3-tiny.conv.15 \
    --config config/yolov3tiny_custom.yaml
```

## MSCOC benchmark

### Setup

1. Download MSCOC 2017 dataset and unzip them.

    ```bash
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    wget http://images.cocodataset.org/zips/train2017.zip
    wget http://images.cocodataset.org/zips/val2017.zip
    unzip train2017.zip
    unzip val2017.zip
    unzip annotations_trainval2017.zip
    ```

    After unzipping, the directory structure should look like

    ```
    <dataset_dir>
    |-- annotations
    |-- train2017
    `-- val2017
    ```

2. Install pycocotools (Cython is required to build pycocotools)

    ```bash
    pip install Cython
    pip install pycocotools
    ```

### Train MSCOCO dataset

```bash
python train_coco.py \
    --dataset_dir /data/COCO \
    --anno_path /data/COCO/annotations/instances_train2017.json \
    --weights weights/darknet53.conv.74 \
    --config config/yolov3_coco.yaml
```

### Evaluate on COCO dataset

```bash
python evaluate_coco.py \
    --dataset_dir /data/COCO \
    --anno_path config/instances_val5k.json \
    --weights weights/yolov3.weights \
    --config config/yolov3_coco.yaml
```

```txt
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.311
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.558
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.313
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.141
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.339
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.457
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.275
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.416
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.437
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.238
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.477
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.603
0.31082299897937926 0.5579624636166972
```
