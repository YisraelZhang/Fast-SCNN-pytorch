from .cityscapes import CitySegmentation
from .datasets.coco import CocoDataset

datasets = {
    'citys': CitySegmentation,
    'coco': CocoDataset,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
