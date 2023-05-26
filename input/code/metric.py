import numpy as np
from detect import get_bboxes, detect


def map_to_bbox(score, geo):
    sample_bboxes = []

    for score_map, geo_map in zip(score, geo):
        bboxes = get_bboxes(score_map, geo_map)
        if bboxes is None:
            bboxes = np.zeros((0,4,2), dtype=np.float32)
        else:
            bboxes = bboxes[:, :8].reshape(-1, 4, 2)
        sample_bboxes.append(bboxes)
    return sample_bboxes