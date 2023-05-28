import numpy as np
from detect import get_bboxes, detect


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