import argparse
from pathlib import Path

import torch
from yolov3.models.yolov3 import YOLOv3
from yolov3.utils import utils as utils
from yolov3.utils.coco_evaluator import COCOEvaluator
from yolov3.utils.parse_yolo_weights import parse_yolo_weights


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--dataset_dir", type=Path, required=True,
                        help="directory path to coco dataset")
    parser.add_argument("--anno_path", type=Path, required=True,
                        help="json filename")
    parser.add_argument("--weights_path", type=Path, default="weights/yolov3.weights",
                        help="path to weights file")
    parser.add_argument("--config_path", type=Path, default="config/yolov3_coco.yaml",
                        help="path to config file")
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help="GPU id to use")
    # fmt: on
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # 設定ファイルを読み込む。
    config = utils.load_config(args.config_path)
    img_size = config["test"]["img_size"]
    batch_size = config["test"]["batch_size"]

    # デバイスを作成する。
    device = utils.get_device(gpu_id=args.gpu_id)

    # モデルを作成する。
    model = YOLOv3(
        config["model"], ignore_threshold=config["train"]["ignore_threshold"]
    )

    if args.weights_path.suffix == ".weights":
        parse_yolo_weights(model, args.weights_path)
        print(f"Darknet format weights file loaded. {args.weights_path}")
    else:
        state = torch.load(args.weights_path)
        model.load_state_dict(state["model_state_dict"])
        print(f"state_dict format weights file loaded. {args.weights_path}")

    model = model.to(device).eval()

    evaluator = COCOEvaluator(
        args.dataset_dir, args.anno_path, img_size=img_size, batch_size=batch_size
    )

    ap50_95, ap50 = evaluator.evaluate(model)
    print(ap50_95, ap50)


if __name__ == "__main__":
    main()
