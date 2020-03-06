import os
import argparse
import time
import shutil

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from data_loader import get_segmentation_dataset
from models.fast_scnn import get_fast_scnn
from utils.loss import MixSoftmaxCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss
from utils.lr_scheduler import LRScheduler
from utils.metric import SegmentationMetric
from mmcv import Config
from data_loader.datasets import build_dataset, build_dataloader
from tools_zym.weight import bbox2multimask

import os
import PIL
import numpy as np

train_dataset = get_segmentation_dataset('coco', split='train', mode='train')
val_dataset = get_segmentation_dataset('coco', split='val', mode='val')

root = os.path.join('./data/mask')

if not os.path.exists(root):
    os.mkdir(root)
    os.mkdir(os.path.join(root, 'train2017'))
    os.mkdir(os.path.join(root, 'val2017'))

# for i, (img, mask) in enumerate(train_dataset):
#     print(mask)
for i, (img, mask) in enumerate(train_dataset):

    ids = train_dataset.img_ids[i]
    image_info = train_dataset.coco.loadImgs(ids)[0]
    imgpath = os.path.join(root, 'train' + '2017',
                           image_info['file_name'])

    mask = transforms.ToPILImage(mode="L")(np.uint8(mask.numpy()))
    mask.save(imgpath, 'JPEG')
    print(np.array(img).shape,
          np.array(mask).shape)
    print(imgpath)

for i, (img, mask) in enumerate(val_dataset):

    ids = val_dataset.img_ids[i]
    image_info = val_dataset.coco.loadImgs(ids)[0]
    imgpath = os.path.join(root, 'val' + '2017',
                           image_info['file_name'])

    mask = transforms.ToPILImage(mode="L")(np.uint8(mask.numpy()))
    mask.save(imgpath, 'JPEG')
    print(imgpath)