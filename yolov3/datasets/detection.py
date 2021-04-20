import albumentations as A
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

import yolov3.utils.utils as utils


class DetectionDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset_dir, train, img_size, bbox_format, augmentation=None,
    ):
        super().__init__()
        self.img_size = img_size
        self.max_labels = 50
        self.init_dataset(dataset_dir, train, bbox_format)
        self.init_augmentation(augmentation, bbox_format)

    def __getitem__(self, index):
        img_path, bboxes, class_ids = self.samples[index]

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
        labels = torch.from_numpy(labels)

        return img, labels, pad_info

    def __len__(self):
        return len(self.samples)

    def init_dataset(self, dataset_dir, train, bbox_format):
        list_path = dataset_dir / "train.txt" if train else "val.txt"
        img_names = open(list_path).read().splitlines()

        classes_path = dataset_dir / "class_names.txt"
        self.class_names = utils.load_classes(classes_path)

        self.samples = []
        for img_name in img_names:
            img_path = dataset_dir / "images" / img_name
            anno_path = dataset_dir / "labels" / f"{img_path.stem}.txt"

            if anno_path.exists():
                bboxes, class_ids = self.load_anno(
                    anno_path, self.class_names, bbox_format
                )
            else:
                bboxes, class_ids = [], []

            self.samples.append((img_path, bboxes, class_ids))

    def load_anno(self, anno_path, class_names, bbox_format):
        lines = open(anno_path).read().splitlines()

        bboxes, class_ids = [], []
        for line in lines:
            label, *coords = line.split()
            coords = list(map(float, coords))
            if bbox_format == "pascal_voc":
                xmin, ymin, xmax, ymax = coords
                x, y, w, h = xmin, ymin, xmax - xmin, ymax - ymin
            elif bbox_format == "coco":
                x, y, w, h = coords
            bboxes.append([x, y, w, h])
            class_ids.append(class_names.index(label))

        return np.array(bboxes), np.array(class_ids)

    def init_augmentation(self, augmentation, bbox_format):
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
                bbox_params=A.BboxParams(format=bbox_format, label_fields=["class_ids"]),
            )
            self.jitter = augmentation["jitter"]
            self.random_placing = augmentation["random_placing"]
        else:
            self.augmentor = None
            self.jitter = 0
            self.random_placing = False