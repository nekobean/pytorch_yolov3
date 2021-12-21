import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def check_det_bboxes(gt_bboxes, det_bboxes, iou_threshold):
    """予測した矩形が正解かどうかを判定する。

    Args:
        gt_bboxes (DataFrame): 正解の矩形一覧。
        det_bboxes (DataFrame): 予測した矩形一覧
        iou_threshold (float): IOU の閾値

    Returns:
        tuple: (正解の矩形一覧, 予測した矩形一覧)
    """
    gt_bboxes = gt_bboxes.copy()
    det_bboxes = det_bboxes.copy()

    # 正解の矩形が予測した矩形と紐付いているかどうかを記録する列を追加する。
    gt_bboxes["Match"] = False
    # 予測した矩形が正解したかどうかを記録する列を追加する。
    det_bboxes["Correct"] = False
    # スコアが高い順にソートする。
    det_bboxes.sort_values("Score", ascending=False, inplace=True)

    for det_bbox in det_bboxes.itertuples():
        # 予測した矩形と同じ画像の正解の矩形一覧を取得する。
        corr_gt_bboxes = gt_bboxes[gt_bboxes["No"] == det_bbox.No]

        if corr_gt_bboxes.empty:
            continue  # この画像に正解の矩形が1つも存在しない場合

        # IOU を計算し、IOU が最大となる正解の矩形を選択する。
        a = np.array([det_bbox.Xmin, det_bbox.Ymin, det_bbox.Xmax, det_bbox.Ymax])
        b = corr_gt_bboxes[["Xmin", "Ymin", "Xmax", "Ymax"]].values
        iou = calc_iou(a, b)

        # 予測した矩形に紐付いた正解の矩形のインデックス
        gt_idx = corr_gt_bboxes.index[iou.argmax()]

        if iou.max() >= iou_threshold and not gt_bboxes.loc[gt_idx, "Match"]:
            # IOU が閾値以上、かつ選択した矩形がまだ他の予測した矩形と紐付いていない場合、正解と判定する。
            gt_bboxes.loc[gt_idx, "Match"] = True
            det_bboxes.loc[det_bbox.Index, "Correct"] = True
            det_bboxes.loc[det_bbox.Index, "IOU"] = iou.max()

    return gt_bboxes, det_bboxes


def calc_pr_curve(gt_bboxes, det_bboxes):
    """閾値ごとの精度及び再現率を計算する。

    Args:
        gt_bboxes (DataFrame): 正解の矩形一覧
        det_bboxes (DataFrame): 予測した矩形一覧

    Returns:
        tuple: (precision, recall)
    """
    TP = det_bboxes["Correct"]
    FP = ~det_bboxes["Correct"]
    n_positives = len(gt_bboxes.index)

    acc_TP = TP.cumsum()
    acc_FP = FP.cumsum()
    precision = acc_TP / (acc_TP + acc_FP)
    recall = acc_TP / n_positives

    return precision, recall


def calc_average_precision(recall, precision):
    """Average Precision (AP) を計算する。

    Args:
        recall (array-like): 閾値ごとの精度一覧
        precision (array-like): 閾値ごとの再現率一覧

    Returns:
        tuple: (修正後の閾値ごとの精度, 修正後の閾値ごとの再現率, AP)
    """
    modified_recall = np.concatenate([[0], recall, [1]])
    modified_precision = np.concatenate([[0], precision, [0]])

    # 末尾から累積最大値を計算する。
    modified_precision = np.maximum.accumulate(modified_precision[::-1])[::-1]

    # AP を計算する。
    average_precision = (np.diff(modified_recall) * modified_precision[1:]).sum()

    return modified_precision, modified_recall, average_precision


def calc_iou(a_bbox, b_bboxes):
    """Calculate intersection over union (IOU).

    Args:
        a (array-like): 矩形の座標を表す形状が (4,) の1次元配列。
        b (array-like): 矩形の座標を表す形状が (NumBoxes, 4) の2次元配列

    Returns:
        (array-like): 矩形 a と b の各矩形の IOU の一覧
    """
    # 短形 a_bbox と短形 b_bboxes の共通部分を計算する。
    xmin = np.maximum(a_bbox[0], b_bboxes[:, 0])
    ymin = np.maximum(a_bbox[1], b_bboxes[:, 1])
    xmax = np.minimum(a_bbox[2], b_bboxes[:, 2])
    ymax = np.minimum(a_bbox[3], b_bboxes[:, 3])
    i_bboxes = np.column_stack([xmin, ymin, xmax, ymax])

    # 矩形の面積を計算する。
    a_area = calc_area(a_bbox)
    b_area = np.apply_along_axis(calc_area, 1, b_bboxes)
    i_area = np.apply_along_axis(calc_area, 1, i_bboxes)

    # IOU を計算する。
    iou = i_area / (a_area + b_area - i_area)

    return iou


def calc_area(bbox):
    """矩形の面積を計算する。

    Args:
        bboxes (array-like): 矩形の座標を表す形状が (4,) の1次元配列。

    Returns:
        float: Areea
    """
    # 矩形の面積を計算する。
    # 共通部分がない場合は、幅や高さは負の値になるので、その場合、幅や高さは 0 とする。
    width = max(0, bbox[2] - bbox[0] + 1)
    height = max(0, bbox[3] - bbox[1] + 1)

    return width * height


def calc_metrics(gt_bboxes, det_bboxes, class_, iou_threshold):
    """指定したクラスの指標を計算する。

    Args:
        gt_bboxes (DataFrame): columns=(No, Label, Xmin, Ymin, Xmax, Ymax)。正解の矩形一覧
        det_bboxes (DataFrame): columns=(No, Score, Label, Xmin, Ymin, Xmax, Ymax)。予測した矩形一覧
        class_ (str): 計算対象のクラス
        iou_threshold (float): IOU の閾値

    Returns:
        dict: 指定したクラスの評価指標
    """
    # 対象クラスの正解及び予測した矩形を抽出する。
    taget_gt_bboxes = gt_bboxes[gt_bboxes["Label"] == class_]
    taget_det_bboxes = det_bboxes[det_bboxes["Label"] == class_]

    # TP, FP を計算する。
    taget_gt_bboxes, taget_det_bboxes = check_det_bboxes(
        taget_gt_bboxes, taget_det_bboxes, iou_threshold
    )

    # PR 曲線を計算する。
    precision, recall = calc_pr_curve(taget_gt_bboxes, taget_det_bboxes)

    modified_precision, modified_recall, average_precision = calc_average_precision(
        recall.values, precision.values
    )

    result = {
        "det_bboxes": taget_det_bboxes,
        "class": class_,
        "precision": precision,
        "recall": recall,
        "modified_precision": modified_precision,
        "modified_recall": modified_recall,
        "average_precision": average_precision,
    }

    return result


def plot_pr_curve(result, save_path):
    """Plot precision recall (PR) cureve.

    Args:
        result (dict): 計算対象のクラス
        save_path (Path, optional): 図を保存するディレクトリ
    """
    fig, ax = plt.subplots()

    ax.grid()
    ax.set_title(
        "Precision Recall Curve\n"
        f"Class: {result['class']}, AP: {result['average_precision']:.2%}"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    # PR 曲線を描画する。
    ax.plot(result["recall"], result["precision"], label="Precision")

    # 修正した PR 曲線を描画する。
    ax.step(
        result["modified_recall"],
        result["modified_precision"],
        "--k",
        label="Modified Precision",
    )

    # 図を保存する。
    fig.savefig(save_path)


def aggregate(results):
    """クラスごとの評価指標を集計する。

    Args:
        results (list of dicts): 各クラスの評価指標の一覧。

    Returns:
        DataFrame: クラスごとの評価指標
    """
    metrics = []
    for result in results:
        metrics.append({"Class": result["class"], "AP": result["average_precision"]})

    metrics = pd.DataFrame(metrics)
    metrics.set_index("Class", inplace=True)
    metrics.sort_index(inplace=True)

    # mAP を計算する。
    metrics.loc["mAP"] = metrics["AP"].mean()

    return metrics
