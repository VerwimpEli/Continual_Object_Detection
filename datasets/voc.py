import os
import random
from typing import Sequence, List, Dict, Any
from copy import deepcopy
from functools import lru_cache

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from xml.etree import ElementTree

VOC_ROOT = '../data/VOC/VOCdevkit/VOC2007/'

VOC_CAT_IDS = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
               "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
               "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

VOC_CAT_SHORT_NAMES = ["plane", "bike", "bird", "boat", "bottle", "bus", "car", "cat",
                       "chair", "cow", "table", "dog", "horse", "motor", "person",
                       "plant", "sheep", "sofa", "train", "tv"]


@lru_cache(maxsize=128)
def get_voc_dict(root_dir: str = VOC_ROOT, split: str = 'train') -> List[Dict[str, Any]]:
    """
    This method should be cached because it's going to be called more than once, and it takes a while.
    """

    train_imgs = os.path.join(root_dir, 'ImageSets/Main', f'{split}.txt')
    train_img_ids = []
    with open(train_imgs) as f:
        for line in f.readlines():
            train_img_ids.append(line.strip('\n'))

    dataset_dicts = []
    for img_id in train_img_ids:
        annot_path = os.path.join(root_dir, 'Annotations', f'{img_id}.xml')
        annot_tree = ElementTree.parse(annot_path)
        annot_root = annot_tree.getroot()

        record = {'file_name': os.path.join(root_dir, 'JPEGImages', annot_root.find('filename').text),
                  'image_id': int(img_id),
                  'height': int(annot_root.find('size').find('height').text),
                  'width': int(annot_root.find('size').find('width').text)}

        objs = []
        for obj in annot_root.findall('object'):
            if obj.find('difficult').text == '1' and split in ['test', 'val']:
                continue
            bbox = obj.find('bndbox')
            objs.append({
                'bbox': [float(bbox.find('xmin').text) - 1, float(bbox.find('ymin').text) - 1,
                         float(bbox.find('xmax').text), float(bbox.find('ymax').text)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": VOC_CAT_IDS.index(obj.find('name').text)
            })

        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def _get_class_incr_voc(root_dir: str, split: str, task_cats: Sequence[int]):
    full_set = get_voc_dict(root_dir, split)
    full_set_copy = deepcopy(full_set)
    for img in full_set_copy:
        img['annotations'] = [annot for annot in img['annotations'] if annot['category_id'] in task_cats]
    full_set_copy = [img for img in full_set_copy if len(img['annotations']) > 0]
    return full_set_copy


def register_class_incremental_voc(classes_per_task: Sequence[Sequence[int]],
                                   names: Sequence[str]):
    thing_classes = VOC_CAT_IDS
    for task, name in zip(classes_per_task, names):
        for split in ['train', 'val', 'trainval', 'test']:
            # Use lambda with default argument, because it should have zero arguments.
            DatasetCatalog.register(f"VOC_{split}_{name}", lambda t=task, s=split: _get_class_incr_voc(VOC_ROOT, s, t))
            MetadataCatalog.get(f"VOC_{split}_{name}").set(thing_classes=thing_classes, dirname=VOC_ROOT, split=split,
                                                           year=2007)


def _get_fine_tune(images_per_task: int):
    full_set = deepcopy(get_voc_dict(VOC_ROOT, 'trainval'))

    random.seed(1997)  # Shuffle set randomly, but always the same.
    random.shuffle(full_set)

    class_count = [0 for _ in range(20)]

    fine_tune_set = []
    for image in full_set:
        for annot in image['annotations']:
            if class_count[annot['category_id']] < images_per_task:
                fine_tune_set.append(image)
                for selected_annot in image['annotations']:
                    class_count[selected_annot['category_id']] += 1
                break

    return fine_tune_set


def register_finetune_voc():
    DatasetCatalog.register("VOC_finetune10", lambda: _get_fine_tune(10))
    MetadataCatalog.get("VOC_finetune10").set(thing_classes=VOC_CAT_SHORT_NAMES, dirname=VOC_ROOT, split='trainval',
                                              year=2007)


def register_voc():
    for d in ["train", "val", "trainval", "test"]:
        DatasetCatalog.register("VOC_" + d, lambda s=d: get_voc_dict(VOC_ROOT, s))
        MetadataCatalog.get("VOC_" + d).set(thing_classes=VOC_CAT_SHORT_NAMES, dirname=VOC_ROOT, split=d, year=2007)


if __name__ == '__main__':
    _get_fine_tune(10)
