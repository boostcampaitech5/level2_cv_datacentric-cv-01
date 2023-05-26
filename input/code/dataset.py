import os.path as osp
import math
import json
from PIL import Image

import torch
import numpy as np
import cv2
import albumentations as A
from torch.utils.data import Dataset
from shapely.geometry import Polygon
from augmentation.augmentation import *

def generate_roi_mask(image, vertices, labels):
    mask = np.ones(image.shape[:2], dtype=np.float32)
    ignored_polys = []
    for vertice, label in zip(vertices, labels):
        if label == 0:
            ignored_polys.append(np.around(vertice.reshape((4, 2))).astype(np.int32))
    cv2.fillPoly(mask, ignored_polys, 0)
    return mask


def filter_vertices(vertices, labels, ignore_under=0, drop_under=0):
    if drop_under == 0 and ignore_under == 0:
        return vertices, labels

    new_vertices, new_labels = vertices.copy(), labels.copy()

    areas = np.array([Polygon(v.reshape((4, 2))).convex_hull.area for v in vertices])
    labels[areas < ignore_under] = 0

    if drop_under > 0:
        passed = areas >= drop_under
        new_vertices, new_labels = new_vertices[passed], new_labels[passed]

    return new_vertices, new_labels


def filter_vertices(vertices, labels, ignore_under=0, drop_under=0):
    if drop_under == 0 and ignore_under == 0:
        return vertices, labels

    new_vertices, new_labels = vertices.copy(), labels.copy()

    areas = np.array([Polygon(v.reshape((4, 2))).convex_hull.area for v in vertices])
    labels[areas < ignore_under] = 0

    if drop_under > 0:
        passed = areas >= drop_under
        new_vertices, new_labels = new_vertices[passed], new_labels[passed]

    return new_vertices, new_labels

class SceneTextDataset(Dataset):
    def __init__(self, root_dir,
                 split='train',
                 image_size=2048,
                 crop_size=1024,
                 ignore_tags=[],
                 ignore_under_threshold=10,
                 drop_under_threshold=1,
                 polygon_masking=True,
                 aug_list=[]):
        with open(osp.join(root_dir, 'ufo/{}.json'.format(split)), 'r') as f:
            anno = json.load(f)

        self.anno = anno
        self.image_fnames = sorted(anno['images'].keys())
        self.image_dir = osp.join(root_dir, 'img', split)

        self.image_size, self.crop_size = image_size, crop_size

        self.ignore_tags = ignore_tags

        self.drop_under_threshold = drop_under_threshold
        self.ignore_under_threshold = ignore_under_threshold

        self.aug_list = aug_list
        self.polygon_masking = polygon_masking
    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        image_fpath = osp.join(self.image_dir, image_fname)

        vertices, labels = [], []
        masking_vertices = []
        for word_info in self.anno['images'][image_fname]['words'].values():
            word_tags = word_info['tags']

            ignore_sample = any(elem for elem in word_tags if elem in self.ignore_tags)
            num_pts = np.array(word_info['points']).shape[0]

            # skip samples with ignore tag and
            # samples with number of points greater than 4
            if ignore_sample :
                continue
            if num_pts > 4 and self.polygon_masking:
                masking_vertices.append(list(map(tuple,word_info['points'])))
                continue
            vertices.append(np.array(word_info['points']).flatten())
            labels.append(int(not word_info['illegibility']))
        vertices, labels = np.array(vertices, dtype=np.float32), np.array(labels, dtype=np.int64)

        vertices, labels = filter_vertices(
            vertices,
            labels,
            ignore_under=self.ignore_under_threshold,
            drop_under=self.drop_under_threshold
        )

        image = Image.open(image_fpath)
        if masking_vertices and self.polygon_masking:
            AnotationMasking(image,masking_vertices)
        if self.aug_list:
            image,vertices,labels=process_augmentation(image,vertices,labels,self.aug_list)
        else:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = np.array(image)
    
        word_bboxes = np.reshape(vertices, (-1, 4, 2))
        roi_mask = generate_roi_mask(image, vertices, labels)

        return image, word_bboxes, roi_mask