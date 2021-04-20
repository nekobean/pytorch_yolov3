import cv2
import torch
import torchvision.transforms as transforms

import yolov3.utils.utils as utils


class Video(torch.utils.data.Dataset):
    def __init__(self, video_path, img_size):
        super().__init__()
        self.cap = cv2.VideoCapture(str(video_path))
        assert self.cap.isOpened(), f"Failed to open video file {video_path}."
        self.img_size = img_size

    def __getitem__(self, index):
        # 画像を読み込む。
        img = self.cap.read()[1]
        # チャンネルの順番を変更する。 (B, G, R) -> (R, G, B)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # レターボックス化する。
        img, pad_info = utils.letterbox(img, self.img_size)
        # PIL -> Tensor
        img = transforms.ToTensor()(img)

        return img, pad_info

    def __len__(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))