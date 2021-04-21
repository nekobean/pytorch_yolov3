#!/bin/bash
cd "$(dirname "$0")"

# Download official YOLOv3 weights trained by MSCOCO.
wget -N https://pjreddie.com/media/files/yolov3.weights

# Download official YOLOv3-tiny weights trained by MSCOCO.
wget -N https://pjreddie.com/media/files/yolov3-tiny.weights
