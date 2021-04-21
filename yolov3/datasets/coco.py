import albumentations as A
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torchvision import transforms as transforms

from yolov3.utils import utils as utils


class COCODataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset_dir, anno_path, img_size, augmentation=None,
    ):
        super().__init__()
        self.img_size = img_size
        self.max_labels = 50
        self.init_dataset(dataset_dir, anno_path)
        # self.init_augmentation(augmentation)
        self.init_augmentation(None)

    def __getitem__(self, index):
        img_id, img_path, bboxes, class_ids = self.samples[index]

        # 画像を読み込む。
        img = cv2.imread(img_path)

        # チャンネルの順番を変更する。 (B, G, R) -> (R, G, B)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.augmentor is not None:
            augmented = self.augmentor(image=img, bboxes=bboxes, class_ids=class_ids)
            img = augmented["image"]
            bboxes = np.array(augmented["bboxes"])
            class_ids = np.array(augmented["class_ids"])

        # 前処理を行う。
        img, pad_info = utils.letterbox(
            img, self.img_size, jitter=self.jitter, random_placing=self.random_placing
        )

        # PIL Image -> Tensor
        img = transforms.ToTensor()(img)

        if len(bboxes) > 0:
            bboxes = utils.coco_to_yolo(bboxes)
            bboxes = utils.encode_bboxes(bboxes, pad_info)
            bboxes /= self.img_size

        # ラベルをパディングする。
        labels = np.column_stack([class_ids, bboxes])  # class_id, x, y, w, h
        labels.resize(self.max_labels, 5)

        # numpy -> Tensor
        labels = torch.from_numpy(labels.astype(np.float32))

        return img, labels, pad_info, img_id

    def __len__(self):
        return len(self.samples)

    def init_dataset(self, dataset_dir, anno_path):
        assert anno_path.exists()
        self.coco = COCO(anno_path)
        self.category_ids = self.coco.getCatIds()

        self.samples = []
        for img_id in self.coco.getImgIds():
            # for img_id in self.coco.getImgIds()[:3]:  # 変更点
            # 画像を読み込む。
            img_name = self.coco.imgs[img_id]["file_name"]
            if (dataset_dir / "train2017" / img_name).exists():
                img_path = str(dataset_dir / "train2017" / img_name)
            else:
                img_path = str(dataset_dir / "val2017" / img_name)

            # 画像 ID のアノテーションを取得する。
            annotations = self.coco.imgToAnns[img_id]
            bboxes = np.array([x["bbox"] for x in annotations])
            class_ids = np.array(
                [self.category_ids.index(x["category_id"]) for x in annotations]
            )

            self.samples.append((img_id, img_path, bboxes, class_ids))

    def init_augmentation(self, augmentation):
        if augmentation:
            transforms = []
            # ランダムに左右反転を行う。
            if augmentation["lr_flip"]:
                transforms.append(A.HorizontalFlip(p=0.5))
            if augmentation["distortion"]:
                transforms.append(
                    A.HueSaturationValue(
                        hue_shift_limit=(-25, 25),
                        sat_shift_limit=(-40, 40),
                        val_shift_limit=(-40, 40),
                        p=0.5,
                    )
                )
            self.augmentor = A.Compose(
                transforms,
                bbox_params=A.BboxParams(format="coco", label_fields=["class_ids"]),
            )
            self.jitter = augmentation["jitter"]
            self.random_placing = augmentation["random_placing"]
        else:
            self.augmentor = None
            self.jitter = 0
            self.random_placing = False
