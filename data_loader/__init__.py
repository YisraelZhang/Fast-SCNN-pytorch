from .cityscapes import CitySegmentation
from .datasets.coco import CocoDataset
from .coco_np import CocoSegmentation

datasets = {
    'citys': CitySegmentation,
    'coco': CocoSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
