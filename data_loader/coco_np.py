"""Ms coco dataloader."""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image, ImageOps, ImageFilter

import skimage.io
import skimage.transform
import skimage.color
import skimage

class CocoSegmentation(Dataset):
    """Ms coco Sementic Segmentation Dataset...."""

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

    NUM_CLASS = 81

    def __init__(self, root='./data/coco', split='train', mode=None, transform=None,
                 base_size=640, crop_size=320, **kwargs):
        super(CocoSegmentation, self).__init__()
        self.root = root
        self.split = split
        self.mode = mode if mode is not None else split
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size


        self.coco = COCO(os.path.join(self.root, 'annotations', 'instances_'+self.split+'2017.json'))
        self.img_ids = self.coco.getImgIds()
        self.img_ids = self.clean(self.img_ids)

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
        return len(self.img_ids)

    def __getitem__(self, item):
        img = self.load_image(item)
        mask = self.load_mask(item)
        #!TODO
        return self._img_transform(img), \
               self._mask_transform(mask)
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask

    def load_image(self, index):
        image_info = self.coco.loadImgs(self.img_ids[index])[0]
        imgpath = os.path.join(self.root, self.split+'2017',
                               image_info['file_name'])
        img = Image.open(imgpath).convert('RGB')
        # if len(np.array(img).shape) < 3:
        #     img = np.stack([img] * 3, axis=2)
        return img

    def load_anns(self, index, img):
        annotation_ids = self.coco.getAnnIds(self.img_ids[index], iscrowd=False)
        # anns is num_anns x 5, (x1, x2, y1, y2, new_idx)
        anns = np.zeros((0, 5))

        # skip the image without annoations
        if len(annotation_ids) == 0:
            H, W, C = np.array(img).shape
            mask = np.zeros((self.NUM_CLASS, H, W))
            mask[0, :, :] = 1
            mask = torch.from_numpy(mask)[None, :, :, :]
            return anns, mask

        coco_anns = self.coco.loadAnns(annotation_ids)
        for a in coco_anns:
            # skip the annotations with width or height < 1
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            ann = np.zeros((1, 5))
            ann[0, :4] = a['bbox']
            ann[0, 4] = self.coco_labels_inverse[a['category_id']]
            anns = np.append(anns, ann, axis=0)

        # (x1, y1, width, height) --> (x1, y1, x2, y2)
        anns[:, 2] += anns[:, 0]
        anns[:, 3] += anns[:, 1]

        H, W, C = np.array(img).shape
        mask = np.zeros((self.NUM_CLASS, H, W))
        mask[0, :, :] = 1
        for i in anns:
            x1, y1, x2, y2, label = i.astype(np.int)
            mask[label+1, y1:y2, x1:x2] = 1
            mask[0, y1:y2, x1:x2] = 0
        mask = torch.from_numpy(mask)[None, :, :, :]

        return anns, mask

    def load_mask(self, index):
        image_info = self.coco.loadImgs(self.img_ids[index])[0]
        maskpath = os.path.join('./data/mask', self.split + '2017',
                                image_info['file_name'])
        mask = Image.open(maskpath)
        # if len(np.array(img).shape) < 3:
        #     img = np.stack([img] * 3, axis=2)
        return mask

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, index):
        image = self.coco.loadImgs(self.img_ids[index])[0]
        return float(image['width']) / float(image['height'])

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = F.interpolate(mask, (oh, ow), mode='nearest')
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask[:, :, y1:y1+outsize, x1:x1+outsize]
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.flip([3])
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = F.interpolate(mask, size=(oh, ow), mode='nearest')
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = F.pad(mask, (0, padw, 0, padh), value=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask[:, :, y1:y1+crop_size, x1:x1+crop_size]
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        # target = mask.squeeze(0)
        _, target = mask.max(dim=1)
        return target.squeeze(0).long()

    def _class_to_index(self, mask):
        H, W, _ = mask.shape
        target = np.zeros((self.NUM_CLASS+1, H, W))
        mask = mask.reshape(-1)
        target = target.reshape(self.NUM_CLASS+1, -1)
        for i, j in enumerate(mask):
            target[i, j] = 1
        target = target.reshape((self.NUM_CLASS+1, H, W))
        target = target[1:, :, :]
        return target

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0

    def clean(self, image_ids):
        for index in range(len(image_ids)):
            image_info = self.coco.loadImgs(image_ids[index])[0]
            imgpath = os.path.join(self.root, self.split+'2017',
                               image_info['file_name'])
            if not os.path.exists(imgpath):
                del image_ids[index]

        return image_ids