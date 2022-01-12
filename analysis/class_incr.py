from collections import Counter
import numpy as np
import argparse
import os
import re
from itertools import chain
from typing import Sequence, Tuple
import logging
import pandas as pd
import tidecv

import datasets as ds

pd.options.display.width = 200
pd.options.display.max_columns = 30


def prepare_paths(dt_folder: str) -> Sequence[Sequence[str]]:
    task_folders = [f for f in os.listdir(dt_folder) if os.path.isdir(os.path.join(dt_folder, f))]
    task_folders = sorted(task_folders, key=lambda x: int(re.findall(r'\d+', x)[-1]))

    print(f"Found {len(task_folders)} tasks: {task_folders}")

    dt_files = []
    for task in task_folders:
        task_files = [os.path.join(dt_folder, task, f)
                      for f in os.listdir(os.path.join(dt_folder, task))
                      if f.endswith('.json')]

        if len(task_files) > 1:
            logging.warning(f"{len(task_files)} json files found in folder {task}, using {task_files[0]}")

        dt_files.append(task_files[0])

    return dt_files


def get_img_ids_of_task(gt_annotations: Sequence, gt_classes: Sequence[int]) -> Sequence[int]:
    """
    Gets all the image ids that have at least a single image of the classes in gt_classes
    :param gt_annotations: List with annotations that have a key 'class' and 'image'
    :param gt_classes: Classes to find images for.
    :return: list of image ids.
    """
    task_ids = set()
    for annot in gt_annotations:
        if annot['class'] in gt_classes:
            task_ids.add(annot['image'])
    return list(task_ids)


def _convert_ids_to_ints(dt_data):
    for annot in dt_data.annotations:
        annot['image'] = int(annot['image'])
    dt_data.images = {int(k): v for k, v in dt_data.images.items()}
    return dt_data


def evaluate_tidecv(gt_path: str, dt_path: str, iou: str = 'voc', gt_classes: Sequence[Sequence[int]] = None) -> \
        Tuple[tidecv.TIDERun, tidecv.TIDE]:
    """
    Runs one evaluation run of the tidecv library.
    :param gt_classes: /
    :param gt_path: path to ground truth file
    :param dt_path: path to detection file
    :param iou: IOU setting to use, voc (0.5) or coco (0.5:0.95)
    :return: The tidecv run, the tide object and the number of gts per class
    """

    tide = tidecv.TIDE()
    gt_data = tidecv.datasets.COCO(gt_path)
    dt_data = tidecv.datasets.COCOResult(dt_path)

    if gt_classes is not None:
        task_img_ids = get_img_ids_of_task(gt_data.annotations, gt_classes)
        gt_classes = set(gt_classes)

        for annot in chain(gt_data.annotations, dt_data.annotations):
            if annot['class'] not in gt_classes or annot['image'] not in task_img_ids:
                annot['ignore'] = True

        # For some reason ignoring alone isn't enough. The gt images should only be those that are used.
        gt_data.images = {k: v for k, v in gt_data.images.items() if k in task_img_ids}

    # Somewhere some img_ids are strings, makes sure they're all ints here.
    dt_data = _convert_ids_to_ints(dt_data)

    # This will use the coco IOU's threshes (if set) for AP calculation, but 0.5 (pos thresh) for the errors
    thresholds = tide.COCO_THRESHOLDS if iou == 'coco' else [0.5]
    tide.evaluate_range(gt_data, dt_data, mode=tide.BOX, thresholds=thresholds, pos_threshold=0.5, name='error')
    return tide.runs['error'], tide


def init_results_df(dataset: str):
    if dataset.lower() == 'voc':
        columns = ds.VOC_CAT_SHORT_NAMES
        indexes = [i for i in range(20)]
    else:
        raise ValueError('Unknown dataset')

    df = pd.DataFrame(columns=['Name', *columns])
    df = df.set_index('Name')
    df.loc['idx'] = indexes
    return df


def extract_class_ap(tide_obj: tidecv.TIDE, results: pd.DataFrame, task_prefix=''):
    classes = results.loc['idx']
    aps = []
    for run in tide_obj.run_thresholds['error']:
        thresh_ap = [run.ap_data.objs[cls].get_ap() for cls in classes]
        thresh_iou = run.pos_thresh
        results.loc[f'{task_prefix}_ap_IOU_{thresh_iou}'] = thresh_ap
        aps.append(thresh_ap)
    aps = np.array(aps)
    results.loc[f'{task_prefix}_mAP'] = np.mean(aps, axis=0)


def extract_ground_truths(tide_obj: tidecv.TIDE, results: pd.DataFrame, task_prefix: str = ''):
    name = f'{task_prefix}_num_gt' if len(task_prefix) > 0 else 'num_gt'
    results.loc[name] = [tide_obj.runs['error'].ap_data.objs[cls].num_gt_positives for cls in results.loc['idx']]


def extract_correct(tide_obj: tidecv.TIDE, results: pd.DataFrame, task_prefix: str = ''):
    run = tide_obj.runs['error']
    used_annotations = [annot['class'] for annot in run.preds.annotations if annot['used']]
    counts = Counter(used_annotations)
    results.loc[f'{task_prefix}_correct'] = [counts[cls] for cls in results.loc['idx']]


def extract_errors(tide_obj: tidecv.TIDE, results: pd.DataFrame, task_prefix: str = ''):
    run = tide_obj.runs['error']
    error_types = ['Cls', 'Loc', 'Miss']

    for error in run.error_dict:
        if error.short_name in error_types:
            error_gt_classes = [err.gt['class'] for err in run.error_dict[error]]
            error_class_counts = Counter(error_gt_classes)
            results.loc[f'{task_prefix}_{error.short_name}_errors'] = [error_class_counts[cls]
                                                                       for cls in results.loc['idx']]


def extract_confusion(tide_obj: tidecv.TIDE, results: pd.DataFrame, task_prefix: str = ''):
    run = tide_obj.runs['error']
    conf_counts = np.zeros((results.shape[1], results.shape[1]))
    for error in run.error_dict[tidecv.ClassError]:
        conf_counts[error.gt['class'], error.pred['class']] += 1

    for i, name in enumerate(results.columns):
        results.loc[f'{task_prefix}_{name}_conf'] = conf_counts[i]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_file')
    parser.add_argument('dt_folder')
    parser.add_argument('--data', default='voc')
    parser.add_argument('--save', action="store_true")
    parser.add_argument('--iou', default='voc', choices=['voc', 'coco'],
                        help='IOU treshold for correct localization, either coco or voc')
    args = parser.parse_args()

    dt_paths = prepare_paths(args.dt_folder)
    results = init_results_df(args.data)

    for i, dt_file in enumerate(dt_paths):
        run, tide = evaluate_tidecv(args.gt_file, dt_file, iou=args.iou)

        extract_class_ap(tide, results, task_prefix=f'T{i + 1}')
        extract_ground_truths(tide, results)
        extract_correct(tide, results, task_prefix=f'T{i + 1}')
        extract_errors(tide, results, task_prefix=f'T{i + 1}')
        extract_confusion(tide, results, task_prefix=f'T{i + 1}')

    print(results)

    if args.save:
        results.to_csv(os.path.join(args.dt_folder, 'class_results.csv'))


if __name__ == '__main__':
    main()
