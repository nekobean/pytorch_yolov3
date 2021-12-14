from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import yaml
from matplotlib import font_manager
from torchvision import ops


def load_config(path):
    """設定ファイルを読み込む。"""
    with open(path) as f:
        config = yaml.safe_load(f)

    return config


def load_classes(path):
    """クラス一覧を読み込む。"""
    with open(path) as f:
        class_names = [x.strip() for x in f.read().splitlines()]

    return class_names


def get_device(gpu_id=-1):
    """Device を取得する。"""
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device("cuda", gpu_id)
    else:
        return torch.device("cpu")


def letterbox(img, img_size, jitter=0, random_placing=False):
    org_h, org_w, _ = img.shape

    if jitter:
        dw = jitter * org_w
        dh = jitter * org_h
        new_aspect = (org_w + np.random.uniform(low=-dw, high=dw)) / (
            org_h + np.random.uniform(low=-dh, high=dh)
        )
    else:
        new_aspect = org_w / org_h

    if new_aspect < 1:
        new_w = int(img_size * new_aspect)
        new_h = img_size
    else:
        new_w = img_size
        new_h = int(img_size / new_aspect)

    if random_placing:
        dx = int(np.random.uniform(img_size - new_w))
        dy = int(np.random.uniform(img_size - new_h))
    else:
        dx = (img_size - new_w) // 2
        dy = (img_size - new_h) // 2

    img = cv2.resize(img, (new_w, new_h))
    pad_img = np.full((img_size, img_size, 3), 127, dtype=np.uint8)
    pad_img[dy : dy + new_h, dx : dx + new_w, :] = img

    scale_x = np.float32(new_w / org_w)
    scale_y = np.float32(new_h / org_h)
    pad_info = (scale_x, scale_y, dx, dy)

    return pad_img, pad_info


def filter_boxes(output, conf_threshold, iou_threshold):
    # 閾値以上の箇所を探す。
    keep_rows, keep_cols = (
        (output[:, 5:] * output[:, 4:5] >= conf_threshold).nonzero().T
    )
    if not keep_rows.nelement():
        return []

    conf_filtered = torch.cat(
        (
            output[keep_rows, :5],
            output[keep_rows, 5 + keep_cols].unsqueeze(1),
            keep_cols.float().unsqueeze(1),
        ),
        1,
    )

    # Non Maximum Suppression を適用する。
    nms_filtered = []
    detected_classes = conf_filtered[:, 6].unique()
    for c in detected_classes:
        detections_class = conf_filtered[conf_filtered[:, 6] == c]
        keep_indices = ops.nms(
            detections_class[:, :4],
            detections_class[:, 4] * detections_class[:, 5],
            iou_threshold,
        )
        detections_class = detections_class[keep_indices]

        nms_filtered.append(detections_class)

    nms_filtered = torch.cat(nms_filtered)

    return nms_filtered


def yolo_to_pascalvoc(bboxes):
    cx, cy, w, h = torch.chunk(bboxes, 4, dim=1)

    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2

    bboxes = torch.cat((x1, y1, x2, y2), dim=1)

    return bboxes


def decode_bboxes(bboxes, info_img):
    scale_x, scale_y, dx, dy = info_img

    bboxes -= torch.stack([dx, dy, dx, dy])
    bboxes /= torch.stack([scale_x, scale_y, scale_x, scale_y])

    return bboxes


def coco_to_yolo(bboxes):
    x, y, w, h = np.split(bboxes, 4, axis=-1)

    cx, cy = x + w / 2, y + h / 2

    bboxes = np.concatenate((cx, cy, w, h), axis=-1)

    return bboxes


def encode_bboxes(bboxes, pad_info):
    scale_x, scale_y, dx, dy = pad_info

    bboxes *= np.array([scale_x, scale_y, scale_x, scale_y])
    bboxes[:, 0] += dx
    bboxes[:, 1] += dy

    return bboxes


def postprocess(outputs, conf_threshold, iou_threshold, pad_infos):
    decoded = []
    for output, *pad_info in zip(outputs, *pad_infos):
        # 矩形の形式を変換する。 (YOLO format -> Pascal VOC format)
        output[:, :4] = yolo_to_pascalvoc(output[:, :4])

        # フィルタリングする。
        output = filter_boxes(output, conf_threshold, iou_threshold)

        # letterbox 処理を行う前の座標に変換する。
        if len(output):
            output[:, :4] = decode_bboxes(output[:, :4], pad_info)

        # デコードする。
        decoded.append(output)

    return decoded


def japanize_matplotlib():
    font_path = str(Path(__file__).parent / "font/ipag.ttc")
    font_manager.fontManager.addfont(font_path)
    matplotlib.rc("font", family="IPAGothic")
