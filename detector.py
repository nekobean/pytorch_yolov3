from pathlib import Path

import torch
from yolov3.datasets.imagefolder import ImageFolder, ImageList
from yolov3.models.yolov3 import YOLOv3, YOLOv3Tiny
from yolov3.utils import utils as utils
from yolov3.utils import vis_utils as vis_utils
from yolov3.utils.parse_yolo_weights import parse_yolo_weights


def output_to_dict(output, class_names):
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


class Detector:
    def __init__(self, config_path, weights_path, gpu_id):
        # 設定ファイルを読み込む。
        self.config = utils.load_config(config_path)
        self.class_names = utils.load_classes(self.config["model"]["class_names"])

        # デバイスを作成する。
        self.device = utils.get_device(gpu_id=gpu_id)

        # モデルを作成する。
        if self.config["model"]["name"] == "yolov3":
            model = YOLOv3(self.config["model"])
        else:
            model = YOLOv3Tiny(self.config["model"])

        # 重みを読み込む。
        if weights_path.suffix == ".weights":
            parse_yolo_weights(model, weights_path)
            print(f"Darknet format weights file loaded. {weights_path}")
        else:
            state = torch.load(weights_path)
            model.load_state_dict(state["model_state_dict"])
            print(f"state_dict format weights file loaded. {weights_path}")

        self.model = model.to(self.device).eval()

    def detect_from_path(self, path):
        img_size = self.config["test"]["img_size"]
        conf_threshold = self.config["test"]["conf_threshold"]
        nms_threshold = self.config["test"]["nms_threshold"]
        batch_size = self.config["test"]["batch_size"]

        # Dataset を作成する。
        dataset = ImageFolder(path, img_size)

        # DataLoader を作成する。
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        # 推論する。
        img_paths, detections = [], []
        for inputs, pad_infos, paths in dataloader:
            inputs = inputs.to(self.device)
            pad_infos = [x.to(self.device) for x in pad_infos]

            with torch.no_grad():
                outputs = self.model(inputs)
                outputs = utils.postprocess(
                    outputs, conf_threshold, nms_threshold, pad_infos
                )

                detections += [output_to_dict(x, self.class_names) for x in outputs]
                img_paths += paths

        return detections, img_paths

    def detect_from_imgs(self, imgs):
        img_size = self.config["test"]["img_size"]
        conf_threshold = self.config["test"]["conf_threshold"]
        nms_threshold = self.config["test"]["nms_threshold"]
        batch_size = self.config["test"]["batch_size"]

        # Dataset を作成する。
        dataset = ImageList(imgs, img_size)

        # DataLoader を作成する。
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        # 推論する。
        detections = []
        for inputs, pad_infos in dataloader:
            inputs = inputs.to(self.device)
            pad_infos = [x.to(self.device) for x in pad_infos]

            with torch.no_grad():
                outputs = self.model(inputs)
                outputs = utils.postprocess(
                    outputs, conf_threshold, nms_threshold, pad_infos
                )

                detections += [output_to_dict(x, self.class_names) for x in outputs]

        return detections

    def detect(self, input):
        if isinstance(input, (str, Path)):
            return self.detect_from_path(input)
        else:
            return self.detect_from_imgs(input)

    def draw_boxes(self, img, detection):
        vis_utils.draw_boxes(img, detection, n_classes=len(self.class_names))
