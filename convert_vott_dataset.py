import argparse
import json
import shutil
from concurrent import futures
from pathlib import Path
from sklearn.model_selection import train_test_split


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="path to source directory")
    parser.add_argument("output", type=Path, help="path to destination directory")
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="the proportion of the dataset to include in the test split",
    )
    args = parser.parse_args()

    return args


def load_label(path):
    # Open label file.
    label = json.load(path.open(encoding="utf8"))

    bboxes = []
    for region in label["regions"]:
        class_name = region["tags"][0]
        xmin = region["boundingBox"]["left"]
        ymin = region["boundingBox"]["top"]
        xmax = region["boundingBox"]["left"] + region["boundingBox"]["width"]
        ymax = region["boundingBox"]["top"] + region["boundingBox"]["height"]

        bboxes.append((class_name, xmin, ymin, xmax, ymax))

    return bboxes


def load_dataset(dataset_dir):
    # Open .vott file.
    vott_path = next(dataset_dir.glob("*.vott"))
    vott = json.load(vott_path.open())

    samples = []
    for asset in vott["assets"].values():
        img_path = dataset_dir / asset["name"]

        # Open label file corresponding to the image if exists.
        label = None
        if asset["state"] == 2:
            label = load_label(dataset_dir / f"{asset['id']}-asset.json")

        samples.append({"image": img_path, "label": label})

    return samples


def write_label(path, label):
    """Write label file.
    """
    with open(path, "w") as f:
        if label is None:
            return  # If the label does not exist, create an empty file.

        for bboxes in label:
            f.write(" ".join(map(str, bboxes)) + "\n")


def output_dataset(output_dir, samples, test_size):
    # Copy images to <output_dir>/images directory.
    imgs_dir = output_dir / "images"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    with futures.ThreadPoolExecutor() as executor:
        for sample in samples:
            src_img_path = sample["image"]
            dst_img_path = imgs_dir / src_img_path.name
            executor.submit(shutil.copy, src_img_path, dst_img_path)

    # Copy labels to <output_dir>/labels directory.
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    with futures.ThreadPoolExecutor() as executor:
        for sample in samples:
            dst_label_path = labels_dir / f"{sample['image'].stem}.txt"
            executor.submit(write_label, dst_label_path, sample["label"])

    # Write train.txt and test.txt.
    train_samples, test_samples = train_test_split(
        samples, test_size=test_size, random_state=42
    )

    with open(output_dir / "train.txt", "w") as f:
        for sample in train_samples:
            f.write(f"{sample['image'].name}\n")

    with open(output_dir / "test.txt", "w") as f:
        for sample in test_samples:
            f.write(f"{sample['image'].name}\n")


def main():
    args = parse_args()

    samples = load_dataset(args.input)
    output_dataset(args.output, samples, args.test_size)


if __name__ == "__main__":
    main()
