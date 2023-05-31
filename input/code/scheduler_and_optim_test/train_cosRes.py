import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from scheduler import CustomScheduler
from model import EAST

from utils import set_seed


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "../data/medical"),
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "trained_models"),
    )

    parser.add_argument("--device", default="cuda" if cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--image_size", type=int, default=2048)
    parser.add_argument("--input_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_epoch", type=int, default=150)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument(
        "--ignore_tags",
        type=list,
        default=["masked", "excluded-region", "maintable", "stamp"],
    )

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError("`input_size` must be a multiple of 32")

    return args


def do_training(
    data_dir,
    model_dir,
    device,
    image_size,
    input_size,
    num_workers,
    batch_size,
    learning_rate,
    max_epoch,
    save_interval,
    ignore_tags,
):
    aug_list = [
        "Resize",
        "AdjustHeight",
        "Rotate",
        "Crop",
        "ToNumpy",
        "ColorJitter",
        "Normalize",
    ]
    dataset = SceneTextDataset(
        data_dir,
        split="train",
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags,
        aug_list=aug_list,
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # CosAnn로 변경
    scheduler_params = {
        "optimizer": optimizer,
        "T_0": max_epoch // 3 + (1 if max_epoch % 3 else 0),
        "T_mult": 1,
        "eta_max": 0.001,
        "T_up": 1,
        "gamma": 0.5,
    }
    # CustomScheduler로 설정
    scheduler = CustomScheduler(
        scheduler_name="CosineAnnealingWarmUpRestarts", params=scheduler_params
    )

    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description("[Epoch {}]".format(epoch + 1))

                loss, extra_info = model.train_step(
                    img, gt_score_map, gt_geo_map, roi_mask
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    "Cls loss": extra_info["cls_loss"],
                    "Angle loss": extra_info["angle_loss"],
                    "IoU loss": extra_info["iou_loss"],
                }
                pbar.set_postfix(val_dict)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]  # 학습률을 확인하기 위해 추가

        print(
            "LR: {} | Mean loss: {:.4f} | Elapsed time: {}".format(
                current_lr,  # 학습률 로그 추가
                epoch_loss / num_batches,
                timedelta(seconds=time.time() - epoch_start),
            )
        )

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, f"{epoch + 1}Epoch.pth")
            torch.save(model.state_dict(), ckpt_fpath)


def main(args):
    torch.cuda.empty_cache()
    seed = 5025
    set_seed(seed)
    do_training(**args.__dict__)


if __name__ == "__main__":
    args = parse_args()
    main(args)