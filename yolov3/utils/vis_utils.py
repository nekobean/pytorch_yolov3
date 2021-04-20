from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from PIL import ImageDraw, ImageFont

font_path = str(Path(__file__).parent / "font/ipag.ttc")


def draw_boxes(img, boxes, n_classes):
    draw = ImageDraw.Draw(img, mode="RGBA")

    # 色を作成する。
    cmap = plt.cm.get_cmap("hsv", n_classes)

    # フォントを作成する。
    fontsize = max(15, int(0.03 * min(img.size)))
    font = ImageFont.truetype(font_path, size=fontsize)

    # クリップする。
    boxes["x1"].clip(0, img.size[0] - 1, inplace=True)
    boxes["y1"].clip(0, img.size[1] - 1, inplace=True)
    boxes["x2"].clip(0, img.size[0] - 1, inplace=True)
    boxes["y2"].clip(0, img.size[1] - 1, inplace=True)

    for box in boxes.itertuples():
        color = tuple(cmap(box.class_id, bytes=True))

        caption = box.class_name
        if hasattr(box, "confidence"):
            caption += f" {box.confidence:.0%}"

        # 矩形を描画する。
        draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=color, width=3)

        # ラベルを描画する。
        text_size = draw.textsize(caption, font=font)
        if box.y1 - text_size[1] >= 0:
            text_origin = np.array([box.x1, box.y1 - text_size[1]])
        else:
            text_origin = np.array([box.x1, box.y1])

        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + text_size - 1)], fill=color
        )
        draw.text(text_origin, caption, fill="black", font=font)
