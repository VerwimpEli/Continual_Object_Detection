import json
import os
from collections import defaultdict
from typing import Dict, Union, List, Callable

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

_HAITAIN_ROOT = '../data/SSLAD-2D/'
HAITAIN_CATS = ["Pedestrian", "Cyclist", "Car", "Truck", "Tram", "Tricycle"]


def get_haitain_dict(root_dir: str = _HAITAIN_ROOT, split: str = 'train'):
    """
    This method loads the original haitain split annotations. They should be located at
    root_dir/labeled/annotations/instance_{split}.json.
    :param root_dir: the root dir of the SSLAD dataset
    :param split: One of train, val or test
    :return: Dictionary with the annotations.
    """
    dict_path = os.path.join(root_dir, 'labeled/annotations/' f'instance_{split}.json')
    with open(dict_path, 'r') as f:
        instances = json.load(f)

    img_annots = defaultdict(list)
    for obj in instances['annotations']:
        obj['category_id'] -= 1
        obj['bbox_mode'] = BoxMode.XYWH_ABS
        img_annots[obj['image_id']].append(obj)

    dataset_dicts = []
    for img in instances['images']:
        img['file_name'] = os.path.join(root_dir, 'labeled', split, img['file_name'])
        img['annotations'] = img_annots[img['id']]
        dataset_dicts.append(img)

    return dataset_dicts


def create_match_fn_from_dict(match_dict: Dict[str, Union[str, List[str]]]) -> \
        Callable[[Dict[str, str]], bool]:
    """
    Create a method that returns true for an image if the given image matches the domain
    specificed by the match dict.
    :param match_dict: The dictionary containin key value pairs which are also present in
    the image annotations.
    :return: The callable method.
    """

    def match_fn(img_annot: Dict[str, str]) -> bool:
        for key, value in match_dict.items():
            if isinstance(value, List):
                if img_annot[key] not in value:
                    return False
            else:
                if img_annot[key] != value:
                    return False
        else:
            return True

    return match_fn


def _get_domain_haitain(root_dir: str, origin_split: str, split: str, task_dict: Dict[str, str],
                        validation_proportion: float = 0.1):
    """
    This shouldn't be called directly, but it filters out the images and annotations that don't match the task
    dictionary. The origin split is the original split of the haitain dataset, while split is whether the
    training or validation data is asked for by the caller of this method.
    """
    full_set = get_haitain_dict(root_dir, origin_split)
    match_fn = create_match_fn_from_dict(task_dict)

    matched_images = [img for img in full_set if match_fn(img)]

    cut_off = int((1.0 - validation_proportion) * len(matched_images))
    if split == "train":
        matched_images = matched_images[:cut_off]
    elif split == "val":
        matched_images = matched_images[cut_off:]
    elif split == 'test':
        pass
    else:
        raise ValueError(f"Unknwon split {split}, should be train or val")

    return matched_images


def register_track_3b_sets(root_dir: str = _HAITAIN_ROOT):
    """
    This method will register the datasets necessary for Track 3B of the SSLAD challenge such that they're available in
    Detectron. Their names are haitain1_[split], haitain2_[split], haitain3_[split] and haitain4_[split], where split
    can be either train, val or test.
    :param root_dir: the root directory
    """

    track_3b_trainval_dicts = \
        [{'city': 'Shanghai', 'location': 'Citystreet', 'period': 'Daytime', 'weather': 'Clear'},
         {'location': 'Highway', 'period': 'Daytime', 'weather': ['Clear', 'Overcast']},
         {'period': 'Night'},
         {'period': 'Daytime', 'weather': 'Rainy'}]
    track_3b_trainval_orig_splits = ['train', 'val', 'val', 'val']
    track_3b_test_dicts = \
        [{'location': ['Citystreet', 'Countryroad'], 'period': 'Daytime', 'weather': ['Clear', 'Overcast']},
         {'location': 'Highway', 'period': 'Daytime', 'weather': ['Clear', 'Overcast']},
         {'period': 'Night'},
         {'period': 'Daytime', 'weather': 'Rainy'}]
    track_3b_names = ['haitain1', 'haitain2', 'haitain3', 'haitain4']

    for task_dict, orig_split, name in zip(track_3b_trainval_dicts, track_3b_trainval_orig_splits,
                                           track_3b_names):
        for split in ['train', 'val']:
            dataset_name = f'{name}_{split}'
            DatasetCatalog.register(dataset_name,
                                    lambda r=root_dir, o=orig_split, s=split, t=task_dict: _get_domain_haitain(r, o, s,
                                                                                                               t))
            MetadataCatalog.get(dataset_name).set(thing_classes=HAITAIN_CATS)

    for task_dict, name in zip(track_3b_test_dicts, track_3b_names):
        dataset_name = f'{name}_test'
        DatasetCatalog.register(dataset_name, lambda r=root_dir, t=task_dict: _get_domain_haitain(r, 'test', 'test', t))
        MetadataCatalog.get(dataset_name).set(thing_classes=HAITAIN_CATS)


if __name__ == '__main__':
    register_track_3b_sets()
