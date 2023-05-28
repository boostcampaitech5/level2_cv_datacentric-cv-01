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
from model import EAST

from utils import set_seed

from torch.utils.data import random_split
from deteval import calc_deteval_metrics
from metric import map_to_bbox
from copy import deepcopy
import numpy as np

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

    parser.add_argument("--image_size", type=int, default=2048)  # resize
    parser.add_argument("--input_size", type=int, default=1024)  # crop
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
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
        "RandomShadow",
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

    val_size = 8
    train_size = len(dataset)- val_size
    train_dataset, val_dataset = random_split(dataset, [train_size,val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )   

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[max_epoch // 2], gamma=0.1
    )

    if not osp.exists(model_dir):
        os.makedirs(model_dir)

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

        print(
            "Mean loss: {:.4f} | Elapsed time: {}".format(
                epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)
            )
        )

        print(f'[Validation {epoch+1}]')
        val_start = time.time()
        with torch.no_grad():
            val_precision=[]
            val_recall=[]
            val_f1 = []

            for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                # get original sizes from image batch
                
                # predict 
                score, geo = model(img.to(device))
                score, geo = deepcopy(score), deepcopy(geo)

                # extract bboxes from score map and geo map
                pred_bboxes = map_to_bbox(score.cpu().numpy(), geo.cpu().numpy())
                gt_score, gt_geo = deepcopy(gt_score_map), deepcopy(gt_geo_map)
                gt_bboxes = map_to_bbox(gt_score.cpu().numpy(), gt_geo.cpu().numpy())

                # calculate metric by deteval with bboxes
                deteval = calc_deteval_metrics(dict(zip([range(len(pred_bboxes))], pred_bboxes)),
                                               dict(zip([range(len(gt_bboxes))], gt_bboxes)))
                
                val_precision.append(deteval['total']['precision'])
                val_recall.append(deteval['total']['recall'])
                val_f1.append(deteval['total']['hmean'])

            precision = np.sum(val_precision) / len(val_precision)
            recall = np.sum(val_recall) / len(val_recall)
            f1 = np.sum(val_f1) / len(val_f1)
            print(f'F1 : {f1:.4f} | Precision : {precision:.4f} | Recall : {recall:.4f}')
            print(f'Validation Elapsed time: {timedelta(seconds=time.time() - val_start)}')

        if epoch == 0:  # loss_record 초기화
            loss_record = epoch_loss / num_batches

        if loss_record > (epoch_loss / num_batches):  # best 모델 저장
            ckpt_fpath = osp.join(model_dir, "best.pth")
            torch.save(model.state_dict(), ckpt_fpath)
            loss_record = epoch_loss / num_batches

        if (epoch + 1) % save_interval == 0:
            ckpt_fpath = osp.join(model_dir, f"epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_fpath)

        if epoch + 1 == max_epoch:  # 마지막 모델 저장
            ckpt_fpath = osp.join(model_dir, "latest.pth")
            torch.save(model.state_dict(), ckpt_fpath)


def main(args):
    # print(args.device)
    seed = 5025
    set_seed(seed)
    do_training(**args.__dict__)


if __name__ == "__main__":
    args = parse_args()
    main(args)
