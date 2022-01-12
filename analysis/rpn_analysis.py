import argparse
import csv
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from detectron2.structures import Boxes, pairwise_iou

from analysis.util import *

plt.style.use('seaborn')


def logit_regression(arr):
    return 1 / (1 + np.exp(-arr))


def summarize_task(obj_scores, tasks):
    task_scores = []
    num_found = []
    for task in tasks:
        task_scores.append([AGGREGATE_FN(obj_scores[idx]) for idx in task])
        num_found.append([len(obj_scores[idx]) for idx in task])

    return task_scores, num_found


def get_gts_by_task(data_dict, tasks):
    cats_all = [annot['category_id'] for img in data_dict.values() for annot in img['annotations']]
    cats_count = Counter(cats_all)
    return [[cats_count[idx] for idx in task] for task in tasks]


def summarize(scores, found, gts_by_task, result_file=None):
    for i, (task_score, task_found, task_gts) in enumerate(zip(scores, found, gts_by_task)):
        nb_found = sum(task_found)
        nb_gts = sum(task_gts)

        print(f"T{i+1}: Found {nb_found} / {nb_gts} objects {nb_found / nb_gts * 100 :.2f}%")
        print(f"T{i+1}: Mean score: {np.mean(task_score):.3f}")
        print(f"T{i+1}: Median score: {np.median(task_score):.3f}")

    if result_file is not None:
        with open(result_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([f'T{1}_found', *[f for task in found for f in task]])
            writer.writerow([f'T{1}_score ({AGGREGATE_FN.__name__})', *[s for task in scores for s in task]])


def plot_scores(scores, gts_by_task, tasks):
    ntasks = len(gts_by_task)
    fig, axes = plt.subplots(1, ntasks, figsize=(7, 2.5))

    for i in range(ntasks):
        task_scores = [score for idx in tasks[i] for score in scores[idx]]
        axes[i].hist(task_scores, bins=2000, cumulative=-1, histtype='stepfilled', linewidth=1,
                     edgecolor=BORDERS[0], facecolor=COLORS[0], alpha=0.9, label=f'After T{i}')

        ratios = [0, 0.25, 0.5, 0.75, 1.0]
        num_objects = sum(gts_by_task[i])
        axes[i].set_ylim(0, 1.05 * num_objects)
        axes[i].set_yticks([r * num_objects for r in ratios])
        axes[i].set_yticklabels(ratios)
        axes[i].set_xlabel('Objectness score')
        axes[i].set_ylabel('Fraction of GT\'s')
        axes[i].set_xlim([0, 1])
        axes[i].set_title(f'T{i} objects')
        axes[i].legend()


def score_analysis(rpn_preds, data_dict, tasks, args):

    num_detections = []
    obj_scores_by_class = defaultdict(list)

    for name, group in rpn_preds:
        group = group[group['score'] > score_thresh]
        num_detections.append(len(group))

        if len(group) == 0:
            continue

        image = data_dict[name]
        gt_boxes = extract_gt_boxes(image, group)
        dt_boxes = extract_dt_boxes(group)

        ious = pairwise_iou(gt_boxes, dt_boxes)

        for i, annot in enumerate(image['annotations']):
            value, idx = torch.max(ious[i], dim=0)
            if value > iou_thresh:
                ious[:, idx] = 0
                cls = annot['category_id']
                obj_scores_by_class[cls].append(group.iloc[idx.item()]['score'])

    del rpn_preds
    task_scores, num_founds = summarize_task(obj_scores_by_class, tasks)

    gts_by_task = get_gts_by_task(data_dict, tasks)
    summarize(task_scores, num_founds, gts_by_task, result_file=args.result_file)
    plot_scores(obj_scores_by_class, gts_by_task, tasks)
    plt.show()


def anchor_box_analysis(rpn_preds, data_dict):

    areas_by_class = defaultdict(list)
    ratios_by_class = defaultdict(list)

    for name, group in rpn_preds:
        image = data_dict[name]
        gt_boxes = extract_gt_boxes(image, group)
        dt_boxes = extract_dt_boxes(group)

        ious = pairwise_iou(gt_boxes, dt_boxes)
        ious, ious_idx = ious.max(dim=1)
        matched_idx = ious_idx[ious > iou_thresh]
        matched_dt_boxes = dt_boxes[matched_idx]

        areas = matched_dt_boxes.area()
        areas = torch.sqrt(areas)

        ratios = (matched_dt_boxes.tensor[:, 2] - matched_dt_boxes.tensor[:, 0]) / \
                 (matched_dt_boxes.tensor[:, 3] - matched_dt_boxes.tensor[:, 1])

        for annot, area, ratio in zip(image['annotations'], areas, ratios):
            cls = annot['category_id']
            areas_by_class[cls].append(area.item())
            ratios_by_class[cls].append(ratio.item())

    fig, axes = plt.subplots(4, 5)
    for i, ax in enumerate(axes.flatten()):
        ax.hist(areas_by_class[i], bins=20)
        ax.set_xlim((0, 800))

    fig, axes = plt.subplots(4, 5)
    for i, ax in enumerate(axes.flatten()):
        ax.hist(ratios_by_class[i], bins=20)
        ax.set_xlim((0, 5))

    plt.show()


def calc_recall(rpn_preds, data_dict, iou_thresholds, max_propsals, task):

    nfound = np.zeros_like(iou_thresholds)
    ngts = 0

    for name, group in rpn_preds:

        image = data_dict[name]
        gt_boxes = extract_task_gt_boxes(image, group, task)
        ngts += len(gt_boxes)
        if len(gt_boxes) == 0:
            continue

        group = group.iloc[:max_propsals]
        dt_boxes = extract_dt_boxes(group)

        ious = pairwise_iou(gt_boxes, dt_boxes)
        ious = ious.max(dim=1).values
        for i, thresh in enumerate(iou_thresholds):
            nfound[i] += sum(ious > thresh).item()

    return nfound / ngts


def average_recall(rpn_preds, data_dict, tasks):
    ious = np.arange(0.5, 1.0, 0.05)
    max_propsals = 1000

    for task in tasks:
        recalls = calc_recall(rpn_preds, data_dict, ious, max_propsals, task)

        print(recalls)
        print(np.mean(recalls))


AGGREGATE_FN = np.mean
COLORS = ['#d62728', '#bcbd22']
BORDERS = ['#93120b', '#93931a']
score_thresh = 0.0
iou_thresh = 0.5


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['area', 'score'], default='score')
    parser.add_argument('--result_file')
    args = parser.parse_args()

    args.result_file = './results/test/class_results.csv'
    dataset = 'voc'
    split = 'test'

    file = './results/test/1/rpn_out.csv'

    tasks = [range(0, 10), range(10, 20)]
    data_dict = get_data_dict(dataset, split)

    rpn_preds = pd.read_csv(file)
    rpn_preds['score'] = logit_regression(rpn_preds['score'])
    rpn_preds = rpn_preds.groupby("image_id")

    average_recall(rpn_preds, data_dict, tasks)
    if args.type == 'score':
        score_analysis(rpn_preds, data_dict, tasks, args)
    else:
        anchor_box_analysis(rpn_preds, data_dict)


if __name__ == '__main__':
    main()
