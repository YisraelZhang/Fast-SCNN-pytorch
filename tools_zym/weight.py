import torch
import torch.nn.functional as F
import numpy as np

import os
import json

def load_weights(img_meta, set_name='weight_linear_train', dir_root='./data/weight'):
    weight_list = []
    pad_shape_list = []
    for i in img_meta:
        pad_shape = i['pad_shape']
        pad_shape_list.append(pad_shape)
    pad_shape = np.array(pad_shape_list).max(axis=0)
    for i in img_meta:
        file_name = i['filename'].split('/')[-1].replace('jpg', 'json')
        img_shape = i['img_shape']
        pad = (0, pad_shape[1] - img_shape[1],
               0, pad_shape[0] - img_shape[0])
        with open(os.path.join(dir_root, set_name, file_name), 'r') as f:
            load_dict = json.load(f)
            weight = json.loads(load_dict)['file_name']
            if len(weight):
                weight = F.interpolate(torch.tensor(weight)[None, None, :, :].float(), size=img_shape[:-1], mode='nearest')
                weight = F.pad(weight, pad=pad, mode='replicate')[0, :, :, :].tolist()
                weight_list.append(weight)
            else:
                weight_list.append(np.ones(pad_shape[:-1])[ None, :, :].tolist())

    weight = torch.tensor(weight_list).cuda()
    return weight

def linear(x=0 ,start=0, end=0):
    if not (end - start):
        end = start + 1
    return (x - start) / (end -start)

def weight_linear(feat_shape, gt_bbox, scale=1.):
    h, w, c = feat_shape
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

def box2weight_linear(ann, img_meta):
    weight_list = []
    for bboxes in ann:
        for bbox in bboxes:
            bbox = bbox.astype(np.int)
            weight = weight_linear(img_meta[0]['img_shape'], bbox, 1.0)
            weight_list.append(weight)

    weight, _ = torch.stack(weight_list, dim=0).max(dim=0)
    return weight

def weight_crop(feat_shape, bbox, scale=1.0):
    H, W, C = feat_shape
    weight = torch.zeros(size=(H, W)).int()
    bbox = torch.tensor(bbox)

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

def weight_multi_crop(feat_shape, bbox, scale=1.0):
    H, W, C = feat_shape
    weight = torch.zeros(size=(H, W)).int()
    bbox = torch.tensor(bbox)

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

def box2weight(ann, img_meta):
    weight_list = []
    for bboxes in ann:
        for bbox in bboxes:
            bbox = bbox.astype(np.int)
            """
            [weight_crop, weight_multi_crop]
            """
            weight = weight_crop(img_meta[0]['img_shape'], bbox, 1.0)
            weight_list.append(weight)
    if not len(weight_list):
        return torch.ones(size=img_meta[0]['img_shape'][:2])
    weight, _ = torch.stack(weight_list, dim=0).max(dim=0)
    return weight

def weight_reshape(weight, img_meta):
    weight_list = []
    pad_shape_list = []
    for i in img_meta:
        pad_shape = i['pad_shape']
        pad_shape_list.append(pad_shape)
    pad_shape = np.array(pad_shape_list).max(axis=0)
    for i in img_meta:
        img_shape = i['img_shape']
        pad = (0, pad_shape[1] - img_shape[1],
               0, pad_shape[0] - img_shape[0])
        if len(weight):
            weight = F.interpolate(torch.tensor(weight)[None, None, :, :].float(), size=img_shape[:-1], mode='nearest')
            weight = F.pad(weight, pad=pad, mode='replicate')[0, :, :, :].tolist()
            weight_list.append(weight)
        else:
            weight_list.append(np.ones(pad_shape[:-1])[ None, :, :].tolist())

    weight = torch.tensor(weight_list).cuda()
    return weight

def det2weight(ann, img_meta):
    """
    convert detection boxes to weight
    :param ann:
    :param img_meta:
    :return:
    """
    """
    [box2weight, box2weight_linear]
    """
    mask = box2weight(ann, img_meta)
    mask = weight_reshape(mask, img_meta)

    return mask

def bbox2multimask(gt_bboxes, gt_labels, img_meta):

    pad_shape = []
    for i in img_meta:
        pad_shape.append(i['pad_shape'])
    pad_shape = torch.tensor(pad_shape).max(dim=0)[0][:2].tolist()
    pad_shape.insert(0, 81)

    mask_list = []
    for i, bboxes in enumerate(gt_bboxes):
        # img_shape = img_meta[i]['img_shape']
        mask = torch.zeros(size=pad_shape)
        mask[0, :, :] = 1
        labels = gt_labels[i]
        for j, bbox in enumerate(bboxes):
            label = labels[j]
            # bbox = random_shift(bbox, 0.3).astype(np.int)
            bbox = bbox.cpu().numpy().astype(np.int)
            x1, y1, x2, y2 = crop_bbox(bbox, img_meta[i])
            mask[label, y1:y2, x1:x2] = 1
            mask[0, y1:y2, x1:x2] = 0
        mask_list.append(mask)

    mask = torch.stack(mask_list).cuda()
    return mask

def crop_bbox(bbox, img_meta):
    img_shape = img_meta['img_shape']
    x1 = np.clip(bbox[0], a_min=0, a_max=img_shape[1])
    x2 = np.clip(bbox[2], a_min=0, a_max=img_shape[1])
    y1 = np.clip(bbox[1], a_min=0, a_max=img_shape[0])
    y2 = np.clip(bbox[3], a_min=0, a_max=img_shape[0])
    return np.array([x1, y1, x2, y2]).astype(np.int)