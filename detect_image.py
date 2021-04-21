import argparse
from pathlib import Path

from PIL import Image
from detector import Detector


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default="data",
        help="image path or directory path which contains images to infer",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="output",
        help="directory path to output detection results",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default="weights/yolov3.weights",
        help="path to weights file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default="config/yolov3_coco.yaml",
        help="path to config file",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    detector = Detector(args.config, args.weights, args.gpu_id)

    # 検出する。
    detections, img_paths = detector.detect(args.input)

    args.output.mkdir(exist_ok=True)
    for detection, img_path in zip(detections, img_paths):
        # 検出結果を出力する。
        print(img_path)
        for box in detection:
            print(
                f"{box['class_name']} {box['confidence']:.0%} ({box['x1']:.0f}, {box['y1']:.0f}, {box['x2']:.0f}, {box['y2']:.0f})"
            )

        # 検出結果を画像に描画して、保存する。
        img = Image.open(img_path)
        detector.draw_boxes(img, detection)
        img.save(args.output / Path(img_path).name)


if __name__ == "__main__":
    main()
