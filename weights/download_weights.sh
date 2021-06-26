#!/bin/bash
cd "$(dirname "$0")"

# Download official YOLOv3 weights trained by MSCOCO.
wget -N https://pjreddie.com/media/files/yolov3.weights

# Download official YOLOv3-tiny weights trained by MSCOCO.
wget -N https://pjreddie.com/media/files/yolov3-tiny.weights

# Download official Darknet 53 weights trained by ImageNet.
wget -N https://pjreddie.com/media/files/darknet53.conv.74
