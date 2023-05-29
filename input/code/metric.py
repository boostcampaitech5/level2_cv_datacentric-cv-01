import os
import numpy as np
from detect import get_bboxes
from PIL import Image, ImageDraw

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

def save_val_result(images, score_maps, geo_maps, pred_bboxes, save_path):
    # for img, score, geo, bboxes in zip(images, score_maps, geo_maps, pred_bboxes):
    for idx in range(images.shape[0]):
        img = (images[idx] * 255).astype(np.uint8)
        img = Image.fromarray(img.transpose(1,2,0))
        
        draw = ImageDraw.Draw(img)
        for bbox in pred_bboxes[idx]:
            points = list(map(tuple, bbox))
            draw.polygon(points, fill=None, outline=(255,0,0), width=1)
        
        img.save(os.path.join(save_path, f"out{idx}.PNG"))

        # score = (score_maps[idx] * 255).astype(np.uint8)
        # score = Image.fromarray(score.transpose(1,2,0))
        # score.save(os.path.join(save_path, f"score.PNG"))

        # geo = (geo_maps[idx] * 255).astype(np.uint8)
        # geo = Image.fromarray(geo.transpose(0, 2, 3, 1))

        # for i in range(5):
        #     geo[i].save(os.path.join(save_path, f"geo{0}.PNG"))
