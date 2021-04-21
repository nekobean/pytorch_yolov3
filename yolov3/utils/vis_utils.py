from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from PIL import ImageDraw, ImageFont

font_path = str(Path(__file__).parent / "font/ipag.ttc")


def get_text_color(color):
    value = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114

    return "black" if value > 128 else "white"


def draw_boxes(img, boxes, n_classes):
    draw = ImageDraw.Draw(img, mode="RGBA")

    # 色を作成する。
    cmap = plt.cm.get_cmap("hsv", n_classes)

    # フォントを作成する。
    fontsize = max(15, int(0.03 * min(img.size)))
    font = ImageFont.truetype(font_path, size=fontsize)

    for box in boxes:
        # 矩形を画像の範囲内にクリップする。
        x1 = int(np.clip(box["x1"], 0, img.size[0] - 1))
        y1 = int(np.clip(box["y1"], 0, img.size[1] - 1))
        x2 = int(np.clip(box["x2"], 0, img.size[0] - 1))
        y2 = int(np.clip(box["y2"], 0, img.size[1] - 1))

        caption = box["class_name"]
        if "confidence" in box:
            caption += f" {box['confidence']:.0%}"

        # 色を選択する。
        color = tuple(cmap(box["class_id"], bytes=True))

        # 矩形を描画する。
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

        # ラベルを描画する。
        text_size = draw.textsize(caption, font=font)
        text_origin = np.array([x1, y1])
        text_color = get_text_color(color)

        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + text_size - 1)], fill=color
        )
        draw.text(text_origin, caption, fill=text_color, font=font)
