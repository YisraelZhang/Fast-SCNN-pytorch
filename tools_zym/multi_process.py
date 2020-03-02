import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from torch.utils.data.dataloader import DataLoader
from .show import show_mask

from .coco import CocoDataset

def imshow(z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = z.shape
    x = np.linspace(0, x, x, endpoint=False)
    y = np.linspace(0, y, y, endpoint=False)
    x, y = np.meshgrid(y, x)
    ax.plot_surface(x, y, z)
    plt.show()

def linear(x=0 ,start=0, end=0):
    if not (end - start):
        end = start + 1
    return (x - start) / (end -start)

def gaus2d(x=0, y=0, mx=0, my=0, sx=2, sy=2):
    sx = torch.clamp_min(sx, 1e-6)
    sy = torch.clamp_min(sy, 1e-6)
    return 1. * torch.exp(-((x - mx) ** 2. / (2. * sx ** 2.) + (y - my) ** 2. / (2. * sy ** 2.)))

def weight_linear(feat, gt_bbox, scale=1.):
    h, w, c = feat.shape
    x = torch.linspace(0, w-1, w)
    y = torch.linspace(0, h-1, h)

    left = linear(x, 0, gt_bbox[0])
    top = linear(y, 0, gt_bbox[1])
    right = linear(x, w, gt_bbox[2])
    down = linear(y, h, gt_bbox[3])

    horizontal = np.clip(np.minimum(left, right), a_min=0,  a_max=1)
    vertical = np.clip(np.minimum(top, down), a_min=0, a_max=1)[:, np.newaxis]

    weight = np.minimum(horizontal, vertical)
    weight[gt_bbox[1]:gt_bbox[3],gt_bbox[0]:gt_bbox[2]] = 1

    return weight

def weight_gaussian(feat, gt_bbox, scale=1.0):

    h, w, c = feat.shape
    x = np.linspace(0, w, w, endpoint=False)
    y = np.linspace(0, h, h, endpoint=False)
    x, y = torch.meshgrid(torch.from_numpy(x), torch.from_numpy(y))

    mx = sum(gt_bbox[::2])/2
    my = sum(gt_bbox[1::2])/2
    sx = (gt_bbox[2] - gt_bbox[0]) * scale
    sy = (gt_bbox[3] - gt_bbox[1]) * scale

    weight = gaus2d(x, y, mx, my, sx, sy)

    return weight

def weight_crop(feat, bbox, scale=1.0):

    H, W, _ = feat.shape
    weight = torch.zeros(size=(H, W)).int()

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    ctr_x = bbox[0] + w / 2
    ctr_y = bbox[1] + h / 2
    x1 = (ctr_x - scale * w / 2).int().clamp_(min=0.0)
    y1 = (ctr_y - scale * h / 2).int().clamp_(min=0.0)
    x2 = (ctr_x + scale * w / 2).int().clamp_(max=W-1)
    y2 = (ctr_y + scale * h / 2).int().clamp_(max=H-1)
    weight[y1:y2, x1:x2] = 1

    return weight

def multi_weight_crop(feat, bbox, scale=1.0):

    H, W, _ = feat.shape
    weight = torch.zeros(size=(H, W)).int()

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    ctr_x = bbox[0] + w / 2
    ctr_y = bbox[1] + h / 2
    if h < 32 and w < 32:
        h *= 1.2
        w *= 1.2
    x1 = (ctr_x - scale * w / 2).int().clamp_(min=0.0)
    y1 = (ctr_y - scale * h / 2).int().clamp_(min=0.0)
    x2 = (ctr_x + scale * w / 2).int().clamp_(max=W-1)
    y2 = (ctr_y + scale * h / 2).int().clamp_(max=H-1)
    weight[y1:y2, x1:x2] = 1

    return weight

def weight_crop_gaussian(feat, bbox, scale=1.0):

    weight = torch.normal(mean=torch.zeros(size=feat.shape[:2]),
                          std=torch.ones(size=feat.shape[:2])).int()
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    ctr_x = bbox[0] + w / 2
    ctr_y = bbox[1] + h / 2
    x1 = (ctr_x - scale * w / 2).int()
    y1 = (ctr_y - scale * h / 2).int()
    x2 = (ctr_x + scale * w / 2).int()
    y2 = (ctr_y + scale * h / 2).int()
    weight[y1:y2, x1:x2] = 1

    return weight

# preprocess
dataset = CocoDataset(set_name='val2017')
classes = dataset.labels
save = dict()
scale =1.0
root = os.path.join('data', 'weight', 'weight_crop_'+str(scale)+'_val')

if not os.path.exists(root):
    os.mkdir(root)

val_loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)

for i, data in enumerate(val_loader):
    ann, img, file_name = data['ann'][0], data['img'][0], data['file_name'][0]
    weight_list = []

    if not len(ann):
        save['file_name'] = []
        js_save = json.dumps(save)
        with open(os.path.join(root, file_name.split('.')[0] + '.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(js_save, ensure_ascii=False))
        print(file_name)
        continue

    for bbox in ann:
        bbox = bbox.int()
        weight = weight_crop(img, bbox, scale)
        # show_mask(weight[None, None, :, :].float())
        # print(classes[bbox[4].item()])
        weight_list.append(weight)

    weight, _ = torch.stack(weight_list, dim=0).max(dim=0)
    # show_mask(weight[None, None, :, :].float())
    save['file_name'] = weight.tolist()

    js_save = json.dumps(save)
    with open(os.path.join(root, file_name.split('.')[0] + '.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(js_save, ensure_ascii=False))
    print(file_name)

js_save = json.dumps(save)
with open('weight_linear_new_val.json','w',encoding='utf-8') as f:
    f.write(json.dumps(js_save,ensure_ascii=False))