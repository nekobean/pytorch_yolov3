import numpy as np
import torch
from torch import nn as nn

from yolov3.utils.utils import bboxes_iou, bboxes_wh_iou


class YOLOLayer(nn.Module):
    def __init__(self, config_model, layer_no, in_ch, ignore_threshold=0.7):
        super().__init__()

        # 特徴マップの1つのセルに対応する元の画像のサイズ
        if config_model["name"] == "yolov3":
            self.stride = [32, 16, 8][layer_no]
        else:
            self.stride = [32, 16][layer_no]

        # -------------------------------------
        self.anch_mask = config_model["anchor_mask"][layer_no]
        self.n_anchors = len(self.anch_mask)
        self.anchors = config_model["anchors"]
        self.all_anchors_grid = [
            (w / self.stride, h / self.stride) for w, h in self.anchors
        ]

        self.masked_anchors = [self.all_anchors_grid[i] for i in self.anch_mask]
        self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
        self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)
        self.ref_anchors = torch.FloatTensor(self.ref_anchors)
        # -------------------------------------

        self.n_classes = config_model["n_classes"]
        self.ignore_thresh = ignore_threshold
        self.l2_loss = nn.MSELoss(reduction="sum")
        self.bce_loss = nn.BCELoss(reduction="sum")
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=self.n_anchors * (self.n_classes + 5),
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def build_taget(pred_boxes, labels):
        pass

    def forward(self, xin, labels=None):
        output = self.conv(xin)
        device = xin.device

        N = output.shape[0]
        A = self.n_anchors
        F = output.shape[2]
        C = 5 + self.n_classes

        y_shift, x_shift = torch.meshgrid(
            torch.arange(F, device=device), torch.arange(F, device=device)
        )
        anchors = torch.tensor(self.masked_anchors, device=device)
        self.ref_anchors = self.ref_anchors.to(device)

        # reshape: (N, A * C, H, W) -> (N, A, C, H, W)
        output = output.reshape(N, A, C, F, F)
        # permute: (N, A, C, H, W) -> (N, A, H, W, C)
        output = output.permute(0, 1, 3, 4, 2)

        # 予測値
        pred_cx = torch.sigmoid(output[..., 0])  # σ(tx)
        pred_cy = torch.sigmoid(output[..., 1])  # σ(ty)
        pred_w = output[..., 2]  # tw
        pred_h = output[..., 3]  # th
        pred_obj = torch.sigmoid(output[..., 4])  # objectness conf
        pred_classes = torch.sigmoid(output[..., 5:])  # class conf

        # 予測の矩形
        pred_boxes = torch.stack(
            [
                pred_cx + x_shift,
                pred_cy + y_shift,
                anchors[:, 0].reshape(1, A, 1, 1) * torch.exp(pred_w),
                anchors[:, 1].reshape(1, A, 1, 1) * torch.exp(pred_h),
            ],
            dim=-1,
        )

        if labels is None:
            output = torch.cat(
                (
                    pred_boxes * self.stride,  #
                    pred_obj.unsqueeze(-1),  #
                    pred_classes,  #
                ),
                dim=-1,
            )
            # (N, A, H, W, 4) -> (N, A * H * W, 5 + n_classes)
            output = output.reshape(N, -1, C)

            return output

        # ラベル
        gb = (
            torch.arange(N, dtype=labels.dtype, device=device).repeat_interleave(
                labels.size(1)
            )
        ).long()
        labels = labels.reshape(-1, 5)

        # 目標値
        obj_mask = torch.zeros(N, A, F, F, device=device, dtype=bool)
        noobj_mask = torch.ones(N, A, F, F, device=device, dtype=bool)
        tx = torch.zeros_like(pred_cx)
        ty = torch.zeros_like(pred_cy)
        tw = torch.zeros_like(pred_w)
        th = torch.zeros_like(pred_h)
        tclasses = torch.zeros_like(pred_classes)
        tgt_scale = torch.zeros_like(pred_w)

        # 正解ラベル
        gt_boxes = labels[:, 1:] * F
        gl = labels[:, 0].long()
        gx = gt_boxes[:, 0]
        gy = gt_boxes[:, 1]
        gw = gt_boxes[:, 2]
        gh = gt_boxes[:, 3]
        gi = gx.long()
        gj = gy.long()

        pred_ious = bboxes_iou(pred_boxes.reshape(-1, 4), gt_boxes)
        pred_best_iou = pred_ious.max(dim=1)[0].reshape(pred_boxes.shape[:-1])
        noobj_mask = pred_best_iou <= self.ignore_thresh  # 閾値未満はつまり物体なしなので True

        # anchor と ラベルを紐付ける 閾値未満でも Anchor Box がマッチしたら物体なしフラグ
        anchor_ious_all = bboxes_wh_iou(gt_boxes, self.ref_anchors)
        best_n_all = torch.max(anchor_ious_all, dim=1)[1]
        best_n = best_n_all % 3
        best_n_mask = (
            (best_n_all == self.anch_mask[0])
            | (best_n_all == self.anch_mask[1])
            | (best_n_all == self.anch_mask[2])
        )

        # no responsible anchor
        noobj_mask[
            gb[best_n_mask], best_n[best_n_mask], gj[best_n_mask], gi[best_n_mask]
        ] = 1
        obj_mask[
            gb[best_n_mask], best_n[best_n_mask], gj[best_n_mask], gi[best_n_mask]
        ] = 1

        # target
        tx[gb, best_n, gj, gi] = gx - gx.floor()
        ty[gb, best_n, gj, gi] = gy - gy.floor()
        tw[gb, best_n, gj, gi] = torch.log(gw / anchors[best_n, 0] + 1e-16)
        th[gb, best_n, gj, gi] = torch.log(gh / anchors[best_n, 1] + 1e-16)
        tgt_scale[gb, best_n, gj, gi] = torch.sqrt(2 - gw * gh / (F * F))
        tobj = obj_mask.float()
        tclasses[gb, best_n, gj, gi, gl] = 1

        tw *= tgt_scale
        th *= tgt_scale
        pred_w *= tgt_scale
        pred_h *= tgt_scale

        # 損失を計算する。
        bceloss = nn.BCELoss(
            weight=tgt_scale[obj_mask] * tgt_scale[obj_mask], reduction="sum",
        )
        loss_x = bceloss(pred_cx[obj_mask], tx[obj_mask])
        loss_y = bceloss(pred_cy[obj_mask], ty[obj_mask])
        loss_w = self.l2_loss(pred_w[obj_mask], tw[obj_mask]) / 2
        loss_h = self.l2_loss(pred_h[obj_mask], th[obj_mask]) / 2
        loss_obj = self.bce_loss(pred_obj[noobj_mask], tobj[noobj_mask])
        loss_cls = self.bce_loss(pred_classes[obj_mask], tclasses[obj_mask])
        loss = loss_x + loss_y + loss_w + loss_h + loss_obj + loss_cls

        return loss, loss_x, loss_y, loss_w, loss_h, loss_obj, loss_cls
