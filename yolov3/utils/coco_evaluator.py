import json
import tempfile

import torch
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from yolov3.datasets.coco import COCODataset
from yolov3.utils.utils import postprocess


class COCOEvaluator:
    def __init__(self, dataset_dir, anno_path, img_size, batch_size):
        # MS COCO の評価を行うときに使用する閾値
        self.conf_threshold = 0.005
        self.nms_threshold = 0.45
        # Dataset を作成する。
        self.dataset = COCODataset(dataset_dir, anno_path, img_size=img_size)
        # DataLoader を作成する。
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size
        )

    def evaluate_results(self, coco, detections):
        tf = tempfile.NamedTemporaryFile(mode="w")
        tf.write(json.dumps(detections))
        img_ids = [x["image_id"] for x in detections]

        cocoDt = coco.loadRes(tf.name)
        cocoEval = COCOeval(coco, cocoDt, "bbox")
        cocoEval.params.imgIds = img_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        ap50_95, ap50 = cocoEval.stats[0], cocoEval.stats[1]

        return ap50_95, ap50

    def output_to_dict(self, output, img_id):
        detection = []
        for x1, y1, x2, y2, obj_conf, class_conf, label in output:
            bbox = {
                "image_id": int(img_id),
                "category_id": self.dataset.category_ids[int(label)],
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(obj_conf * class_conf),
            }

            detection.append(bbox)

        return detection

    def evaluate(self, model):
        model.eval()
        device = next(model.parameters()).device

        detections = []
        for inputs, _, pad_infos, img_ids in tqdm(self.dataloader, desc="infer"):
            inputs = inputs.to(device)
            pad_infos = [x.to(device) for x in pad_infos]
            with torch.no_grad():
                # 順伝搬を行う。
                outputs = model(inputs)
                # 後処理を行う。
                outputs = postprocess(
                    outputs, self.conf_threshold, self.nms_threshold, pad_infos
                )

            for output, img_id in zip(outputs, img_ids):
                detections += self.output_to_dict(output, img_id)

        if len(detections) > 0:
            ap50_95, ap50 = self.evaluate_results(self.dataset.coco, detections)
        else:
            ap50_95, ap50 = 0, 0

        return ap50_95, ap50
