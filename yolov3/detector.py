from pathlib import Path

import torch
from yolov3.datasets.imagefolder import ImageFolder, ImageList
from yolov3.datasets.video import Video
from yolov3.utils import utils as utils
from yolov3.utils import vis_utils as vis_utils
from yolov3.utils.model import create_model, parse_yolo_weights


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
    def __init__(self, config_path, weights_path, gpu_id=0):
        config_path = Path(config_path)
        weights_path = Path(weights_path)

        # 設定ファイルを読み込む。
        self.config = utils.load_config(config_path)
        self.class_names = utils.load_classes(
            config_path.parent / self.config["model"]["class_names"]
        )

        # Device を作成する。
        self.device = utils.get_device(gpu_id=gpu_id)

        # モデルを作成する。
        model = create_model(self.config)
        if weights_path.suffix == ".weights":
            parse_yolo_weights(model, weights_path)
            print(f"Darknet format weights file loaded. {weights_path}")
        else:
            state = torch.load(weights_path)
            model.load_state_dict(state["model"])
            print(f"Checkpoint file {weights_path} loaded.")
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

    def detect_from_video(self, path):
        img_size = self.config["test"]["img_size"]
        conf_threshold = self.config["test"]["conf_threshold"]
        nms_threshold = self.config["test"]["nms_threshold"]
        batch_size = self.config["test"]["batch_size"]

        # Dataset を作成する。
        dataset = Video(path, img_size)

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

    def draw_boxes(self, img, detection):
        vis_utils.draw_boxes(img, detection, n_classes=len(self.class_names))
