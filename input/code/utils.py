import torch
import random
import os
import numpy as np
import torch.backends.cudnn as cudnn
import wandb

from detect import get_bboxes
from PIL import Image, ImageDraw

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def map_to_bbox(score, geo):
    """extract bboxes from score map and geo map

    Args:
        score (torch.Tensor): score map
        geo (torch.Tensor): geometric map 
        org_sizes (list): list of image original sizes 
        map_scale (float, optional): used for map margin. Defaults to 0.5.

    Returns:
        sample_bboxes : final bboxes <list>
    """
    sample_bboxes = []

    for score_map, geo_map in zip(score, geo):
        bboxes = get_bboxes(score_map, geo_map)
        if bboxes is None:
            bboxes = np.zeros((0,4,2), dtype=np.float32)
        else:
            bboxes = bboxes[:, :8].reshape(-1, 4, 2)
        sample_bboxes.append(bboxes)
    return sample_bboxes

def save_val_result(images, pred_bboxes, save_path):
    """save valid prediction images and log 

    Args:
        images (numpy.array): images
        pred_bboxes (list): predicted bboxes
        save_path (str): path to save images
    """
    for idx in range(images.shape[0]):
        img = images[idx]
        img = img - np.min(img)
        img = (img / np.max(img) * 255).astype(np.uint8)
        img = np.clip(img, 0, 255)
        img = Image.fromarray(img.transpose(1,2,0))
        
        draw = ImageDraw.Draw(img)
        for bbox in pred_bboxes[idx]:
            points = list(map(tuple, bbox))
            draw.polygon(points, fill=None, outline=(255,0,0), width=1)
        
        img.save(os.path.join(save_path, f"out{idx}.PNG"))
        wandb.log({f"out{idx}":wandb.wandb.Image(img, caption=idx)})