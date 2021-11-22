import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from yolov3.detector import Detector


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument(
        "--input", type=Path, default="data",
        help="image path or directory path which contains images to infer",
    )
    parser.add_argument(
        "--output", type=Path, default="output",
        help="directory path to output detection results",
    )
    parser.add_argument(
        "--weights", type=Path, default="weights/yolov3.weights",
        help="path to weights file",
    )
    parser.add_argument(
        "--config", type=Path, default="config/yolov3_coco.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0,
        help="GPU id to use")
    # fmt: on
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    detector = Detector(args.config, args.weights, args.gpu_id)

    # 検出する。
    print("Detecting video...")
    detections = detector.detect_from_video(args.input)

    args.output.mkdir(exist_ok=True)
    output_path = args.output / f"{args.input.stem}.avi"
    print(f"Writing video to {output_path}...")

    # VideoCapture を作成する。
    cap = cv2.VideoCapture(str(args.input))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # VideoWriter を作成する。
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for detection in detections:
        # 1フレームずつ取得する。
        frame = cap.read()[1]

        # OpenCV -> Pillow
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        detector.draw_boxes(frame, detection)

        # Pillow -> OpenCV
        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # フレームを書き込む。
        writer.write(frame)

    writer.release()
    cap.release()


if __name__ == "__main__":
    main()
