import json
import os
from copy import deepcopy
from detectron2.data.datasets import register_coco_instances
from functools import lru_cache
from typing import List, Dict, Any


# The annotations should be under this root folder: root/annotations/instances_[split][year].json
COCO_ROOT = "../data/COCO"

# The images should be in a sub folder of this path, called [split][year]. So for 2014 train the images should
# be under image_root/train2014/
COCO_IMAGE_ROOT = f"{COCO_ROOT}/images"


@lru_cache(maxsize=128)
def get_coco_dict(root_dir: str = COCO_ROOT, split: str = 'train', year: int = 2014) -> List[Dict[str, Any]]:

    assert split in ['train', 'val'], f"Unknown split {split}"
    annot_file = os.path.join(root_dir, 'annotations', f'instances_{split}{year}.json')

    with open(annot_file, 'r') as f:
        data = json.load(f)

    return data


def _register_ci_coco(root_dir: str, split: str, task: int,  year: int = 2014):
    """
    :param root_dir:
    :param split: either train or val
    :param task: Task 1 signifies the first 40 classes, task 2 will return the next 40. Like this, because
    class_ids aren't contigous.
    :param year: 2014 by default.

    Note: this will cache annotation files, deleting them will recreate them.
    """

    name_postfix = f'{split}{year}_task_{task}'
    annot_path = os.path.join(root_dir, 'annotations', f'instances_{name_postfix}.json')

    if not os.path.isfile(annot_path):
        full_set = get_coco_dict(root_dir, split, year)
        full_set_copy = deepcopy(full_set)

        split_cat_id = 46  # TODO: Hardcode for now.
        if task == 0:
            full_set_copy['annotations'] = [annot for annot in full_set_copy['annotations']
                                            if annot['category_id'] < split_cat_id]
        else:
            full_set_copy['annotations'] = [annot for annot in full_set_copy['annotations']
                                            if annot['category_id'] >= split_cat_id]

        imgs_with_annotations = set([annot['image_id'] for annot in full_set_copy['annotations']])
        full_set_copy['images'] = [img for img in full_set_copy['images'] if img['id'] in imgs_with_annotations]

        with open(annot_path, 'w') as f:
            json.dump(full_set_copy, f)
    else:
        print("[INFO] Using previously created task annotation file. Delete if this is unwanted.")

    register_coco_instances(f'COCO_{name_postfix}', {}, annot_path, os.path.join(COCO_IMAGE_ROOT, f"{split}{year}"))


def register_ci_coco():
    _register_ci_coco(COCO_ROOT, 'train', 0)
    _register_ci_coco(COCO_ROOT, 'train', 1)
    _register_ci_coco(COCO_ROOT, 'val', 0)
    _register_ci_coco(COCO_ROOT, 'val', 1)
