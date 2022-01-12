import torch
from detectron2.structures import Boxes

import datasets as ds


def get_data_dict(dataset='voc', split='test'):
    if dataset == 'voc':
        train_dict = ds.get_voc_dict(split=split)
    else:
        raise ValueError(f"Dataset {dataset} unkown or not implemented")

    return {img['image_id']: img for img in train_dict}


def rescale_boxes(gt_boxes, img_meta, group):
    width, height = group.iloc[0][['width', 'height']]
    orig_width, orig_height = img_meta['width'], img_meta['height']
    gt_boxes.scale(width / orig_width, height / orig_height)


def extract_gt_boxes(img_meta, group):
    gt_boxes = [annot['bbox'] for annot in img_meta['annotations']]
    gt_boxes = Boxes(torch.tensor(gt_boxes))
    rescale_boxes(gt_boxes, img_meta, group)
    return gt_boxes


def extract_task_gt_boxes(img_meta, group, task):
    gt_boxes = [annot['bbox'] for annot in img_meta['annotations'] if annot['category_id'] in task]
    gt_boxes = Boxes(torch.tensor(gt_boxes))
    rescale_boxes(gt_boxes, img_meta, group)
    return gt_boxes


def extract_dt_boxes(group):
    return Boxes(torch.tensor(group[['x1', 'y1', 'x2', 'y2']].to_numpy()))

