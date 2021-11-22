import argparse
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from yolov3.datasets.custom import CustomDataset
from yolov3.utils import utils as utils
from yolov3.utils.model import create_model, parse_yolo_weights


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", type=Path, required=True,
        help="directory path to custom dataset",
    )
    parser.add_argument(
        "--weights", type=Path, default="weights/darknet53.conv.74",
        help="path to darknet weights file (.weights) or checkpoint file (.pth)",
    )
    parser.add_argument(
        "--config", type=Path, default="config/yolov3_custom.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0,
        help="GPU id to use")
    parser.add_argument(
        "--save_dir", type=Path, default="train_output",
        help="directory where checkpoint files are saved",
    )
    parser.add_argument(
        "--save_interval", type=int, default=1000,
        help="interval between saving checkpoints",
    )
    # fmt: on
    args = parser.parse_args()

    return args


def build_optimizer(config, model):
    batch_size = config["train"]["batch_size"]
    subdivision = config["train"]["subdivision"]
    n_samples_per_iter = batch_size * subdivision  # 1反復あたりのサンプル数
    momentum = config["train"]["momentum"]
    decay = config["train"]["decay"]
    base_lr = config["train"]["lr"] / n_samples_per_iter

    params = []
    for key, value in model.named_parameters():
        weight_decay = decay * n_samples_per_iter if "conv.weight" in key else 0
        params.append({"params": value, "weight_decay": weight_decay})

    optimizer = torch.optim.SGD(
        params,
        lr=base_lr,
        momentum=momentum,
        weight_decay=decay * n_samples_per_iter,
    )

    return optimizer


def build_scheduler(config, optimizer):
    burn_in = config["train"]["burn_in"]
    steps = config["train"]["steps"]

    def schedule(i):
        if i < burn_in:
            factor = (i / burn_in) ** 4
        elif i < steps[0]:
            factor = 1.0
        elif i < steps[1]:
            factor = 0.1
        else:
            factor = 0.01

        return factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    return scheduler


def repeater(dataloader):
    for loader in repeat(dataloader):
        for data in loader:
            yield data


def get_train_size():
    return np.random.randint(10, 20) * 32


def print_info(info, max_iter):
    print(f"iteration: {info['iter']}/{max_iter}", end=" ")
    print(f"image size: {info['img_size']}", end=" ")
    print(f"[Loss] total: {info['loss_total']:.2f}", end=" ")
    print(f"xy: {info['loss_xy']:.2f}", end=" ")
    print(f"wh: {info['loss_wh']:.2f}", end=" ")
    print(f"object: {info['loss_obj']:.2f}", end=" ")
    print(f"class: {info['loss_cls']:.2f}")


def main():
    args = parse_args()

    # 設定ファイルを読み込む。
    config = utils.load_config(args.config)
    img_size = config["train"]["img_size"]
    batch_size = config["train"]["batch_size"]
    subdivision = config["train"]["subdivision"]
    n_samples_per_iter = batch_size * subdivision
    max_iter = config["train"]["max_iter"]
    random_size = config["augmentation"]["random_size"]
    class_names = utils.load_classes(
        args.config.parent / config["model"]["class_names"]
    )

    # Device を作成する。
    device = utils.get_device(gpu_id=args.gpu_id)

    # --weights にチェックポイントが指定された場合は読み込む。
    state = torch.load(args.weights) if args.weights.suffix == ".ckpt" else None

    # モデルを作成する。
    model = create_model(config)
    if state:
        # --weights に指定されたファイルがチェックポイントの場合、状態を復元する。
        model.load_state_dict(state["model_state_dict"])
        print(f"Checkpoint file {args.weights} loaded.")
    else:
        # --weights に指定されたファイルが Darknet 形式の重みの場合、重みを読み込む。
        parse_yolo_weights(model, args.weights)
        print(f"Darknet weights file {args.weights} loaded.")
    model = model.to(device).train()

    # Optimizer を作成する。
    optimizer = build_optimizer(config, model)
    iter_state = 1

    # Scheduler を作成する。
    scheduler = build_scheduler(config, optimizer)

    if state:
        # --weights に指定されたファイルがチェックポイントの場合、状態を復元する。
        # Optimizer の状態を読み込む。
        optimizer.load_state_dict(state["optimizer_state_dict"])
        # Scheduler の状態を読み込む。
        scheduler.load_state_dict(state["scheduler_state_dict"])
        # 前回のステップの次のステップから再開する
        iter_state = state["iter"] + 1

    # Dataset を作成する。
    train_dataset = CustomDataset(
        args.dataset_dir,
        class_names,
        train=True,
        img_size=img_size,
        bbox_format="pascal_voc",
        augmentation=config["augmentation"],
    )

    # DataLoader を作成する。
    train_dataloader = repeater(
        torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    )

    # チェックポイントを保存するディレクトリを作成する。
    args.save_dir.mkdir(exist_ok=True)

    history = []
    for iter_i in range(iter_state, max_iter + 1):
        optimizer.zero_grad()
        for _ in range(subdivision):
            imgs, targets, _ = next(train_dataloader)
            imgs = imgs.to(device)
            targets = targets.to(device)

            # 順伝搬する。
            loss = model(imgs, targets)

            # 逆伝搬する。
            loss.backward()

        optimizer.step()
        scheduler.step()

        # 学習過程を記録する。
        info = {
            "iter": iter_i,
            "lr": scheduler.get_last_lr()[0] * n_samples_per_iter,
            "img_size": train_dataset.img_size,
            "loss_total": float(loss),
            "loss_xy": float(model.loss_dict["xy"]),
            "loss_wh": float(model.loss_dict["wh"]),
            "loss_obj": float(model.loss_dict["obj"]),
            "loss_cls": float(model.loss_dict["cls"]),
        }
        history.append(info)
        print_info(info, max_iter)

        # モデルの入力サイズを変更する。
        if random_size and iter_i % 10 == 0:
            train_dataset.img_size = get_train_size()
            print(f"input image size changed to {train_dataset.img_size}.")

        if iter_i % args.save_interval == 0 or iter_i == max_iter:
            model_name = config["model"]["name"]
            if iter_i == max_iter:
                state_dict = {"model_state_dict": model.state_dict()}
                state_save_path = args.save_dir / f"{model_name}_final.pth"
                history_save_path = args.save_dir / "history_final.csv"
            else:
                state_dict = {
                    "iter": iter_i,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }
                state_save_path = args.save_dir / f"{model_name}_{iter_i:06d}.ckpt"
                history_save_path = args.save_dir / f"history_{iter_i:06d}.csv"

            # チェックポイントを保存する。
            torch.save(state_dict, state_save_path)

            # 学習経過を保存する。
            pd.DataFrame(history).to_csv(history_save_path, index=False)

            print(
                f"Training state saved. checkpoints: {state_save_path}, loss history: {history_save_path}."
            )


if __name__ == "__main__":
    main()
