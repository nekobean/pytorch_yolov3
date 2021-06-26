import numpy as np
import torch
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


def bboxes_iou(bboxes_a, bboxes_b):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = torch.max(
        (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
        (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
    )
    # bottom right
    br = torch.min(
        (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
        (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
    )

    area_a = torch.prod(bboxes_a[:, 2:], 1)
    area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def bboxes_iou2(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        # bottom right
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
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

        ################################################################################
        # ref anchors は消す
        self.ref_anchors = np.zeros((len(self.all_anchors), 4))
        self.ref_anchors[:, 2:] = np.array(self.all_anchors)
        self.l2_loss = nn.MSELoss(reduction="sum")
        self.bce_loss = nn.BCELoss(reduction="sum")

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

    def calc(self, xin, output, labels):
        ref_anchors = torch.FloatTensor(self.ref_anchors)
        N = output.shape[0]
        A = self.n_anchors
        F = output.shape[2]
        C = self.n_classes + 5

        # reshape: (N, A * C, H, W) -> (N, A, C, H, W)
        output = output.reshape(N, A, C, F, F)
        # permute: (N, A, C, H, W) -> (N, A, H, W, C)
        output = output.permute(0, 1, 3, 4, 2)

        y_shift, x_shift = torch.meshgrid(
            torch.arange(F, dtype=xin.dtype, device=xin.device),
            torch.arange(F, dtype=xin.dtype, device=xin.device),
        )
        anchors = torch.tensor(self.anchors, device=xin.device)
        w_anchors = anchors[:, 0].reshape(1, A, 1, 1)
        h_anchors = anchors[:, 1].reshape(1, A, 1, 1)

        ####################################

        dtype = torch.cuda.FloatTensor if xin.is_cuda else torch.FloatTensor

        # logistic activation for xy, obj, cls
        output[..., np.r_[:2, 4:C]] = torch.sigmoid(output[..., np.r_[:2, 4:C]])

        # calculate pred - xywh obj cls
        pred = output.clone()
        pred[..., 0] += x_shift
        pred[..., 1] += y_shift
        pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors
        pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors

        if labels is None:
            pred[..., :4] *= self.stride
            return pred.reshape(N, -1, C)

        pred = pred[..., :4].data

        # target assignment

        tgt_mask = torch.zeros(N, A, F, F, 4 + self.n_classes).type(dtype)
        obj_mask = torch.ones(N, A, F, F).type(dtype)
        tgt_scale = torch.zeros(N, A, F, F, 2).type(dtype)
        target = torch.zeros(N, A, F, F, C).type(dtype)

        labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = labels[:, :, 1] * F
        truth_y_all = labels[:, :, 2] * F
        truth_w_all = labels[:, :, 3] * F
        truth_h_all = labels[:, :, 4] * F
        truth_i_all = truth_x_all.to(torch.int16).numpy()
        truth_j_all = truth_y_all.to(torch.int16).numpy()

        for b in range(N):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = dtype(np.zeros((n, 4)))
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou2(truth_box.cpu(), ref_anchors)
            best_n_all = np.argmax(anchor_ious_all, axis=1)
            best_n = best_n_all % 3
            best_n_mask = (
                (best_n_all == self.anchor_indices[0])
                | (best_n_all == self.anchor_indices[1])
                | (best_n_all == self.anchor_indices[2])
            )

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou2(pred[b].reshape(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = pred_best_iou > self.ignore_threshold
            pred_best_iou = pred_best_iou.reshape(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = 1 - pred_best_iou.type(torch.int16)

            if sum(best_n_mask) == 0:
                continue

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(
                        torch.int16
                    ).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(
                        torch.int16
                    ).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / anchors[best_n[ti], 0] + 1e-16
                    )
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / anchors[best_n[ti], 1] + 1e-16
                    )
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 0].to(torch.int16).numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(
                        2 - truth_w_all[b, ti] * truth_h_all[b, ti] / F / F
                    )

        output[..., 4] *= obj_mask
        output[..., np.r_[0:4, 5:C]] *= tgt_mask
        output[..., 2:4] *= tgt_scale

        target[..., 4] *= obj_mask
        target[..., np.r_[0:4, 5:C]] *= tgt_mask
        target[..., 2:4] *= tgt_scale

        bceloss = nn.BCELoss(weight=tgt_scale * tgt_scale, size_average=False)
        loss_xy = bceloss(output[..., :2], target[..., :2])
        loss_wh = self.l2_loss(output[..., 2:4], target[..., 2:4]) / 2
        loss_obj = self.bce_loss(output[..., 4], target[..., 4])
        loss_cls = self.bce_loss(output[..., 5:], target[..., 5:])
        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, target

    def calc2(self, xin, predictions, labels):
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
        wh_weight = torch.zeros(nB, nA, nG, nG, 2, dtype=xin.dtype, device=xin.device)

        obj_mask = torch.ones(nB, nA, nG, nG, dtype=xin.dtype, device=xin.device)
        noobj_mask = torch.zeros(
            nB, nA, nG, nG, 4 + self.n_classes, dtype=xin.dtype, device=xin.device
        )

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

                wh_weight[b, a, j, i] = torch.sqrt(
                    2 - gt_w[b, n] * gt_h[b, n] / (nG * nG)
                )
                obj_mask[b, a, j, i] = 1
                noobj_mask[b, a, j, i] = 1

        # 損失計算の対象外をマスクする。
        predictions[..., 4] *= obj_mask
        target[..., 4] *= obj_mask
        predictions[..., np.r_[:4, 5:nC]] *= noobj_mask
        target[..., np.r_[:4, 5:nC]] *= noobj_mask

        predictions[..., 2:4] *= wh_weight
        target[..., 2:4] *= wh_weight

        bceloss = nn.BCELoss(weight=wh_weight * wh_weight, reduction="sum")
        loss_xy = bceloss(predictions[..., :2], target[..., :2])
        loss_wh = self.l2_loss(predictions[..., 2:4], target[..., 2:4]) / 2
        loss_obj = self.bce_loss(predictions[..., 4], target[..., 4])
        loss_cls = self.bce_loss(predictions[..., 5:], target[..., 5:])
        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, target

    def forward(self, xin, labels=None):
        output = self.conv(xin)
        return self.calc2(xin, output, labels)
        #########################
        output = self.conv(xin)
        if labels is None:
            return self.calc2(xin, output.clone(), labels)

        if True:
            loss, loss_xy, loss_wh, loss_obj, loss_cls, target = self.calc(
                xin, output.clone(), labels
            )
            (loss2, loss_xy2, loss_wh2, loss_obj2, loss_cls2, target2,) = self.calc2(
                xin, output.clone(), labels
            )
            assert torch.allclose(loss, loss2)
            assert torch.allclose(loss_xy, loss_xy2)
            assert torch.allclose(loss_wh, loss_wh2), f"{loss_wh} vs {loss_wh2}"
            assert torch.allclose(loss_obj, loss_obj2)
            assert torch.allclose(loss_cls, loss_cls2)
            return loss, loss_xy, loss_wh, loss_obj, loss_cls

        ##############################
        import time

        start = time.time()
        for i in range(100):
            loss, loss_xy, loss_wh, loss_obj, loss_cls, target = self.calc(
                xin, output.clone(), labels
            )
        print(time.time() - start)

        start = time.time()
        for i in range(100):
            (loss2, loss_xy2, loss_wh2, loss_obj2, loss_cls2, target2,) = self.calc2(
                xin, output.clone(), labels
            )
        print(time.time() - start)

        assert loss == loss2, f"{loss} {loss2}"
        assert loss_xy == loss_xy2
        assert loss_wh == loss_wh2
        assert loss_obj == loss_obj2
        assert loss_cls == loss_cls2
        assert torch.allclose(target, target2)
        exit(1)

        return loss, loss_xy, loss_wh, loss_obj, loss_cls
