import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from yolov3.datasets.video import Video
from yolov3.models.yolov3 import YOLOv3, YOLOv3Tiny
from yolov3.utils import utils as utils
from yolov3.utils import vis_utils as vis_utils
from yolov3.utils.parse_yolo_weights import parse_yolo_weights


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


def output_to_frame(output, class_names):
    detection = []
    for x1, y1, x2, y2, obj_conf, class_conf, label in output:
        bbox = {
            "confidence": float(obj_conf * class_conf),
            "class_id": int(label),
            "class_name": class_names[int(label)],
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
        }
        detection.append(bbox)

    return detection


def output_detections(input_path, output_path, detections, n_classes):
    # VideoCapture を作成する。
    cap = cv2.VideoCapture(str(input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # VideoWriter を作成する。
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for detection in tqdm(detections, desc="output"):
        # 1フレームずつ取得する。
        frame = cap.read()[1]

        # OpenCV -> Pillow
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        vis_utils.draw_boxes(frame, detection, n_classes)

        # Pillow -> OpenCV
        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # フレームを書き込む。
        writer.write(frame)

    writer.release()
    cap.release()


def main():
    args = parse_args()

    # 設定ファイルを読み込む。
    config = utils.load_config(args.config)
    class_names = utils.load_classes(config["model"]["class_names"])
    img_size = config["test"]["img_size"]
    conf_threshold = config["test"]["conf_threshold"]
    nms_threshold = config["test"]["nms_threshold"]
    batch_size = config["test"]["batch_size"]

    # デバイスを作成する。
    device = utils.get_device(gpu_id=args.gpu_id)

    # Dataset を作成する。
    dataset = Video(args.input, img_size)

    # DataLoader を作成する。
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # モデルを作成する。
    if config["model"]["name"] == "yolov3":
        model = YOLOv3(config["model"])
    else:
        model = YOLOv3Tiny(config["model"])

    if args.weights.suffix == ".weights":
        parse_yolo_weights(model, args.weights)
        print(f"Darknet format weights file loaded. {args.weights}")
    else:
        state = torch.load(args.weights)
        model.load_state_dict(state["model_state_dict"])
        print(f"state_dict format weights file loaded. {args.weights}")

    model = model.to(device).eval()

    # 推論する。
    detections = []
    for inputs, pad_infos in tqdm(dataloader, desc="infer"):
        inputs = inputs.to(device)
        pad_infos = [x.to(device) for x in pad_infos]

        with torch.no_grad():
            outputs = model(inputs)
            outputs = utils.postprocess(
                outputs, conf_threshold, nms_threshold, pad_infos
            )

            detections += [output_to_frame(x, class_names) for x in outputs]

    # 検出結果を出力する。
    args.output.mkdir(exist_ok=True)
    output_path = args.output / f"{args.input.stem}.avi"
    output_detections(args.input, output_path, detections, len(class_names))


if __name__ == "__main__":
    main()
