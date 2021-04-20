import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils import data as data
from tqdm import tqdm
from yolov3.datasets.imagefolder import ImageFolder
from yolov3.models.yolov3 import YOLOv3, YOLOv3Tiny
from yolov3.utils import utils as utils
from yolov3.utils import vis_utils as vis_utils
from yolov3.utils.parse_yolo_weights import parse_yolo_weights


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--input_path", type=Path, default="data",
                        help="image path or directory path which contains images to infer")
    parser.add_argument("--output_dir", type=Path, default="output",
                        help="directory path to output detection results")
    parser.add_argument("--weights_path", type=Path, default="weights/yolov3.weights",
                        help="path to weights file")
    parser.add_argument("--config_path", type=Path, default="config/yolov3_coco.yaml",
                        help="path to config file")
    parser.add_argument('--gpu_id', type=int, default=0,
                        help="GPU id to use")
    # fmt: on
    args = parser.parse_args()

    return args


def output_to_frame(output, class_names):
    detection = []
    for x1, y1, x2, y2, obj_conf, class_conf, label in output:
        bbox = {
            "confidence": float(obj_conf * class_conf),
            "class_id": int(label),
            "class_name": class_names[int(label)],
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
        }
        detection.append(bbox)
    detection = pd.DataFrame(detection)

    return detection


def output_detection(output_dir, img_path, detection, n_classes):
    output_dir.mkdir(exist_ok=True)

    # 検出結果を出力する。
    print(img_path)
    print(detection)

    # 検出結果を画像に描画して、保存する。
    img = Image.open(img_path)
    vis_utils.draw_boxes(img, detection, n_classes)
    img.save(output_dir / Path(img_path).name)


def main():
    args = parse_args()

    # 設定ファイルを読み込む。
    config = utils.load_config(args.config_path)
    class_names = utils.load_classes(config["model"]["class_names"])
    img_size = config["test"]["img_size"]
    conf_threshold = config["test"]["conf_threshold"]
    nms_threshold = config["test"]["nms_threshold"]
    batch_size = config["test"]["batch_size"]

    # デバイスを作成する。
    device = utils.get_device(gpu_id=args.gpu_id)

    # Dataset を作成する。
    dataset = ImageFolder(args.input_path, img_size)

    # DataLoader を作成する。
    dataloader = data.DataLoader(dataset, batch_size=batch_size)

    # モデルを作成する。
    if config["model"]["name"] == "yolov3":
        model = YOLOv3(config["model"])
    else:
        model = YOLOv3Tiny(config["model"])

    if args.weights_path.suffix == ".weights":
        parse_yolo_weights(model, args.weights_path)
        print(f"Darknet format weights file loaded. {args.weights_path}")
    else:
        state = torch.load(args.weights_path)
        model.load_state_dict(state["model_state_dict"])
        print(f"state_dict format weights file loaded. {args.weights_path}")

    model = model.to(device).eval()

    # 検出を行う。
    import time

    start = time.time()
    img_paths, detections = [], []
    for inputs, pad_infos, paths in tqdm(dataloader, desc="infer"):
        inputs = inputs.to(device)
        pad_infos = [x.to(device) for x in pad_infos]

        with torch.no_grad():
            outputs = model(inputs)
            outputs = utils.postprocess(
                outputs, conf_threshold, nms_threshold, pad_infos
            )

            detections += [output_to_frame(x, class_names) for x in outputs]
            img_paths += paths

            import numpy as np

            for path, output in zip(paths, outputs):
                out_path = Path("test") / Path(path).with_suffix(".npy").name
                assert np.allclose(
                    np.array(output[:, :4].cpu()),
                    np.load(out_path),
                    rtol=1e-3,
                    atol=1e-3,
                )

    avg_time = (time.time() - start) / len(img_paths)
    print(f"Average inference time: {avg_time:.3f} s/image")

    # 検出結果を出力する。
    args.output_dir.mkdir(exist_ok=True)
    for img_path, detection in zip(img_paths, detections):
        output_detection(args.output_dir, img_path, detection, len(class_names))


if __name__ == "__main__":
    main()
