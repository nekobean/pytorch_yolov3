import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F


def bboxes_iou_wh(size_a: torch.Tensor, size_b: torch.Tensor):
    """2つの大きさの IOU を計算する。

    Args:
        size_a (torch.Tensor): 矩形の大きさ
        size_b (torch.Tensor): 矩形の大きさ

    Returns:
        torch.Tensor: 2つの矩形の大きさの IOU を計算する。
    """
    area_a = size_a.prod(1)
    area_b = size_b.prod(1)
    area_i = torch.min(size_a[:, None], size_b).prod(2)

    return area_i / (area_a[:, None] + area_b - area_i)


def bboxes_iou(bboxes_a: torch.Tensor, bboxes_b: torch.Tensor):
    """2つの矩形の IOU を計算する。

    Args:
        bboxes_a (torch.Tensor): [description]
        bboxes_b (torch.Tensor): [description]

    Returns:
        [type]: [description]
    """
    area_a = torch.prod(bboxes_a[:, 2:], 1)
    area_b = torch.prod(bboxes_b[:, 2:], 1)

    tl = torch.max(
        (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
        (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
    )
    br = torch.min(
        (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
        (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
    )

    area_i = (br - tl).prod(2) * (tl < br).all(2)

    return area_i / (area_a[:, None] + area_b - area_i)


class YOLOLayer(nn.Module):
    def __init__(self, config: dict, layer_no: int, in_ch: int):
        """
        Args:
            config (dict): モデルの設定
            layer_no (int): YOLO レイヤーのインデックス
            in_ch (int): 入力チャンネル数

        Attributes:
            stride (int): グリッドに対応する元の画像の画素数
            n_classes (int): クラス数
            ignore_threshold (float): 無視する IOU の閾値
            all_anchors (list): すべての anchor box 一覧 (グリッド座標系)
            anchor_indices (list): このレイヤーで使用する anchor box のインデックスの一覧
            anchors (list): このレイヤーで使用する anchor box の一覧 (グリッド座標系)
            n_anchors (int): このレイヤーの anchor box の数
        """
        super().__init__()
        if config["name"] == "yolov3":
            self.stride = [32, 16, 8][layer_no]
        else:
            self.stride = [32, 16][layer_no]

        self.n_classes = config["n_classes"]
        self.ignore_threshold = config["ignore_threshold"]
        self.all_anchors = [
            (w / self.stride, h / self.stride) for w, h in config["anchors"]
        ]
        self.anchor_indices = config["anchor_mask"][layer_no]
        self.anchors = [self.all_anchors[i] for i in self.anchor_indices]
        self.n_anchors = len(self.anchor_indices)
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=self.n_anchors * (self.n_classes + 5),
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def get_anchor_indices(self, gt_bboxes: torch.Tensor, anchors: torch.Tensor):
        """ground truth に対応する anchor box のインデックスを取得する。

        Args:
            gt_bboxes (torch.Tensor): ground truth の矩形一覧
            anchors (torch.Tensor): anchor box の一覧

        Returns:
            [type]: 最も IOU が高い anchor box の一覧
        """
        # grount truth の矩形の大きさと anchor box の大きさとの IOU を計算する。
        anchor_ious = bboxes_iou_wh(gt_bboxes, anchors)

        # 最も IOU が高い anchor box のインデックスを取得する。
        best_anchor_indices = torch.max(anchor_ious, dim=1)[1]

        # このレイヤーの anchor box でない場合はインデックスを -1 にする。
        mask = (
            (best_anchor_indices == self.anchor_indices[0])
            | (best_anchor_indices == self.anchor_indices[1])
            | (best_anchor_indices == self.anchor_indices[2])
        )
        best_anchor_indices = torch.where(mask, best_anchor_indices % 3, -1)

        return best_anchor_indices

    def calc(self, xin, predictions, labels):
        nB = predictions.size(0)  # バッチサイズ
        nA = self.n_anchors  # anchor box の数
        nG = predictions.size(2)  # 特徴マップのサイズ
        nC = self.n_classes + 5  # チャンネル数

        # reshape: (N, A * C, H, W) -> (N, A, C, H, W) -> (N, A, H, W, C)
        predictions = predictions.reshape(nB, nA, nC, nG, nG).permute(0, 1, 3, 4, 2)

        anchors = torch.tensor(self.anchors, dtype=xin.dtype, device=xin.device)
        all_anchors = torch.tensor(self.all_anchors, dtype=xin.dtype, device=xin.device)

        # 予測値
        x = torch.sigmoid(predictions[..., 0])
        y = torch.sigmoid(predictions[..., 1])
        w = predictions[..., 2]
        h = predictions[..., 3]
        pred_obj = torch.sigmoid(predictions[..., 4])
        pred_cls = torch.sigmoid(predictions[..., 5:])

        y_shift, x_shift = torch.meshgrid(
            torch.arange(nG, dtype=xin.dtype, device=xin.device),
            torch.arange(nG, dtype=xin.dtype, device=xin.device),
        )
        pred_boxes = torch.stack(
            [
                x + x_shift,
                y + y_shift,
                torch.exp(w) * anchors[:, 0].reshape(1, nA, 1, 1),
                torch.exp(h) * anchors[:, 1].reshape(1, nA, 1, 1),
            ],
            dim=-1,
        )

        output = torch.cat(
            (pred_boxes * self.stride, pred_obj.unsqueeze(-1), pred_cls), -1,
        ).reshape(nB, -1, nC)

        if labels is None:
            return output

        # logistic activation for xy, obj, cls
        predictions[..., np.r_[:2, 4:nC]] = torch.sigmoid(
            predictions[..., np.r_[:2, 4:nC]]
        )

        target = torch.zeros(nB, nA, nG, nG, nC, dtype=xin.dtype, device=xin.device)
        scale = torch.zeros(nB, nA, nG, nG, 1, dtype=xin.dtype, device=xin.device)
        obj_mask = torch.ones(nB, nA, nG, nG, dtype=xin.dtype, device=xin.device)
        noobj_mask = torch.zeros(nB, nA, nG, nG, 1, dtype=xin.dtype, device=xin.device)

        gt_cls = labels[:, :, 0].long()
        gt_boxes = labels[:, :, 1:] * nG
        gt_x = gt_boxes[:, :, 0]
        gt_y = gt_boxes[:, :, 1]
        gt_w = gt_boxes[:, :, 2]
        gt_h = gt_boxes[:, :, 3]
        gi = gt_boxes[:, :, 0].long()
        gj = gt_boxes[:, :, 1].long()

        for b in range(nB):
            n_bboxes = (labels[b].sum(1) > 0).sum()
            if n_bboxes == 0:
                continue

            # ground truth に対応する anchor box のインデックスを取得する。
            anchor_indices = self.get_anchor_indices(
                gt_boxes[b, :n_bboxes, 2:], all_anchors
            )

            # iou > ignore_threshold となる位置を False にする
            pred_ious = bboxes_iou(pred_boxes[b].reshape(-1, 4), gt_boxes[b, :n_bboxes])
            pred_best_iou = pred_ious.max(dim=1)[0]
            pred_best_iou = pred_best_iou <= self.ignore_threshold
            obj_mask[b] = pred_best_iou.reshape(pred_boxes[b].shape[:3])

            for n in range(n_bboxes):
                if anchor_indices[n] == -1:
                    continue

                a = anchor_indices[n]
                i, j = gi[b, n], gj[b, n]

                target[b, a, j, i, 0] = gt_x[b, n] - gt_x[b, n].floor()
                target[b, a, j, i, 1] = gt_y[b, n] - gt_y[b, n].floor()
                target[b, a, j, i, 2] = torch.log(gt_w[b, n] / anchors[a, 0] + 1e-16)
                target[b, a, j, i, 3] = torch.log(gt_h[b, n] / anchors[a, 1] + 1e-16)
                target[b, a, j, i, 4] = 1
                target[b, a, j, i, 5 + gt_cls[b, n]] = 1

                scale[b, a, j, i] = 2 - gt_w[b, n] * gt_h[b, n] / (nG * nG)
                obj_mask[b, a, j, i] = 1
                noobj_mask[b, a, j, i] = 1

        # 損失計算の対象外をマスクする。
        predictions[..., 4] *= obj_mask
        target[..., 4] *= obj_mask
        predictions[..., np.r_[:4, 5:nC]] *= noobj_mask
        target[..., np.r_[:4, 5:nC]] *= noobj_mask

        loss_xy = F.binary_cross_entropy(
            predictions[..., :2], target[..., :2], weight=scale, reduction="sum"
        )
        loss_wh = (
            F.mse_loss(
                predictions[..., 2:4] * torch.sqrt(scale),
                target[..., 2:4] * torch.sqrt(scale),
                reduction="sum",
            )
            * 0.5
        )
        loss_obj = F.binary_cross_entropy(
            predictions[..., 4], target[..., 4], reduction="sum"
        )
        loss_cls = F.binary_cross_entropy(
            predictions[..., 5:], target[..., 5:], reduction="sum"
        )
        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, target

    def forward(self, xin, labels=None):
        output = self.conv(xin)
        return self.calc(xin, output, labels)
