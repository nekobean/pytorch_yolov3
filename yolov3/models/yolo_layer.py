import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn


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
            gt_bboxes (torch.Tensor): (N, 2) Tensor。ground truth の大きさ一覧。
            anchors (torch.Tensor): (N, 2) Tensor。anchor box の一覧。

        Returns:
            torch.Tensor: 最も IOU が高い anchor box のインデックス。
        """
        # ground truth の矩形の大きさと anchor box の大きさとの IOU を計算する。
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

    def calc(self, x, labels):
        nB = x.size(0)  # バッチサイズ
        nA = self.n_anchors  # anchor box の数
        nG = x.size(2)  # 特徴マップの縦横のグリッド数
        nC = self.n_classes + 5  # チャンネル数

        anchors = torch.tensor(self.anchors, dtype=x.dtype, device=x.device)
        all_anchors = torch.tensor(self.all_anchors, dtype=x.dtype, device=x.device)

        # (N, A * C, H, W) -> (N, A, C, H, W) -> (N, A, H, W, C)
        x = x.reshape(nB, nA, nC, nG, nG).permute(0, 1, 3, 4, 2)

        # シグモイド関数を適用する。
        x[..., np.r_[:2, 4:nC]] = torch.sigmoid(x[..., np.r_[:2, 4:nC]])

        # 矩形の中心及び大きさを計算する。
        y_offset, x_offset = torch.meshgrid(
            torch.arange(nG, dtype=x.dtype, device=x.device),
            torch.arange(nG, dtype=x.dtype, device=x.device),
            indexing="ij",
        )

        pred_boxes = torch.stack(
            [
                x[..., 0] + x_offset,
                x[..., 1] + y_offset,
                torch.exp(x[..., 2]) * anchors[:, 0].reshape(1, nA, 1, 1),
                torch.exp(x[..., 3]) * anchors[:, 1].reshape(1, nA, 1, 1),
            ],
            dim=-1,
        )

        if labels is None:
            # 推論時の場合、矩形の情報は画像座標系にして返す。
            output = torch.cat((pred_boxes * self.stride, x[..., 4:]), dim=-1).reshape(
                nB, -1, nC
            )
            return output

        # 正解データ
        target = torch.zeros(nB, nA, nG, nG, nC, dtype=x.dtype, device=x.device)
        # tx, ty, tw, th の損失計算時の重み
        scale = torch.zeros(nB, nA, nG, nG, 1, dtype=x.dtype, device=x.device)
        # objectness score の損失計算対象を表すマスク。
        obj_mask = torch.ones(nB, nA, nG, nG, dtype=torch.bool, device=x.device)
        # objectness score 以外の損失計算対象を表すマスク。
        noobj_mask = torch.zeros(nB, nA, nG, nG, 1, dtype=torch.bool, device=x.device)

        gt_cls = labels[:, :, 0].long()
        gt_boxes = labels[:, :, 1:] * nG
        gt_x = gt_boxes[:, :, 0]
        gt_y = gt_boxes[:, :, 1]
        gt_w = gt_boxes[:, :, 2]
        gt_h = gt_boxes[:, :, 3]
        gi = gt_boxes[:, :, 0].long()
        gj = gt_boxes[:, :, 1].long()

        for b in range(nB):
            n_bboxes = (labels[b].sum(dim=1) > 0).sum()  # この画像のラベル数
            if n_bboxes == 0:
                continue  # 1つもラベルがない場合

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

                scale[b, a, j, i] = 2 - gt_w[b, n] * gt_h[b, n] / nG ** 2
                obj_mask[b, a, j, i] = True
                noobj_mask[b, a, j, i] = True

        # 損失計算の対象外をマスクする。
        x[..., 4] *= obj_mask
        target[..., 4] *= obj_mask
        x[..., np.r_[:4, 5:nC]] *= noobj_mask
        target[..., np.r_[:4, 5:nC]] *= noobj_mask

        loss_xy = F.binary_cross_entropy(
            x[..., :2], target[..., :2], weight=scale, reduction="sum"
        )

        # 入力及びラベルを \sqrt{scale} 倍すると、勾配が scale 倍される。
        # darknet の実装に合わせて、1/2 倍する。
        loss_wh = (
            F.mse_loss(
                x[..., 2:4] * torch.sqrt(scale),
                target[..., 2:4] * torch.sqrt(scale),
                reduction="sum",
            )
            * 0.5
        )

        loss_obj = F.binary_cross_entropy(x[..., 4], target[..., 4], reduction="sum")
        loss_cls = F.binary_cross_entropy(x[..., 5:], target[..., 5:], reduction="sum")
        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls

    def forward(self, xin, labels=None):
        output = self.conv(xin)
        return self.calc(output, labels)
