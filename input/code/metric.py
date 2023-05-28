import numpy as np
from detect import get_bboxes, detect


def map_to_bbox(score, geo, input_size, org_sizes, map_scale=0.5):
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

    for score_map, geo_map, org_size in zip(score, geo, org_sizes):
        map_margin = int(abs(org_size[0] - org_size[1]) * map_scale * input_size / max(org_size))

        if org_size[0] == org_size[1]:
            score_map, geo_map = score_map, geo_map
        elif org_size[0] > org_size[1]:
            score_map, geo_map = score_map[:, :, :-map_margin], geo_map[:, :, :-map_margin]
        else:
            score_map, geo_map = score_map[:, :-map_margin, :], geo_map[:, :-map_margin, :]

        bboxes = get_bboxes(score_map, geo_map)
        if bboxes is None:
            bboxes = np.zeros((0,4,2), dtype=np.float32)
        else:
            bboxes = bboxes[:, :8].reshape(-1, 4, 2)
            bboxes *= max(org_size) / input_size
        sample_bboxes.append(bboxes)
    return sample_bboxes