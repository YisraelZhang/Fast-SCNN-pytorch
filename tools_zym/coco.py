import os
import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage

import cv2

import random

class CocoDataset(Dataset):
    """
    simple coco dataset.
    """
    def __init__(self,
                 dir='./data/coco',
                 set_name='val2017',):
        """

        :param dir:
        :param set_name:
        """
        self.root_dir = dir
        self.set_name = set_name

        self.coco = COCO(os.path.join(dir, 'annotations/instances_'+set_name+'.json'))
        self.image_ids = self.coco .getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        # coco ids is not from 1, and not continue
        # make a new index from 0 to 79, continuely

        # classes:              {names:     new_index}
        # coco_labels:          {new_idex:  coco_index}
        # coco_labels_inverse   {coco_idex: new_index}
        self.classes, self.coco_labels, self.coco_labels_inverse = {}, {}, {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # labels:               {new_index: names}
        self.labels = {}
        for k, v in self.classes.items():
            self.labels[v] = k

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        index = self.image_ids[item]
        img, file_name = self.load_image(index)
        ann =  self.load_anns(index)
        return dict(img=img, ann=ann, file_name=file_name)

    def load_image(self, index):

        img_info = self.coco.loadImgs(index)
        img_path = os.path.join(self.root_dir, self.set_name, img_info[0]['file_name'])

        # img = skimage.io.imread(img_path)[:, :, ::-1]
        img = cv2.imread(img_path)

        return img, img_info[0]['file_name']

    def load_anns(self, index):

        ann_ids = self.coco.getAnnIds(index, iscrowd=False)

        anns = np.ones((0, 5))

        if len(ann_ids) == 0:
            return anns

        coco_anns = self.coco.loadAnns(ann_ids)

        for a in coco_anns:
            # # skip the annotations with width or height < 1
            # if a['bbox'][2] < 1 or a['bbox'][3] < 1:
            #     continue

            ann = np.zeros((1, 5))
            ann[0, :4] = a['bbox']
            ann[0, 4] = self.coco_labels_inverse[a['category_id']]
            anns = np.append(anns, ann, axis=0)

        anns[:, 2] += anns[:, 0]
        anns[:, 3] += anns[:, 1]

        return anns