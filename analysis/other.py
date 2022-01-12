import argparse
import matplotlib.pyplot as plt
import tidecv

from class_incr import evaluate_tidecv
from datasets import VOC_CAT_SHORT_NAMES


def plot_prec_rec_curves(run: tidecv.TIDERun):
    classes = len(run.ap_data.objs)

    fig, axes = plt.subplots(classes // 5, 5)
    for i, ax in enumerate(axes.flatten()):
        ap_obj = run.ap_data.objs[i]
        ax.plot(ap_obj.curve[0], ap_obj.curve[1])
        ax.set_title(VOC_CAT_SHORT_NAMES[i])


def plot_class_error_confidence(run: tidecv.TIDERun):
    class_errors = run.error_dict[tidecv.ClassError]
    scores = [err.pred['score'] for err in class_errors]
    scores = sorted(scores, reverse=True)
    plt.plot(scores)


def main():
    gt_file = '../../detectron/results/VOC_test_coco_format_nodiff.json'
    dt_file = '../results/test/1/coco_instances_results.json'

    all20 = '/home/eli/Documents/Doctoraat/code/detectron/results/all20/0/coco_instances_results.json'

    for file in [dt_file, all20]:
        run, tide = evaluate_tidecv(gt_file, file, iou='voc')
        # plot_prec_rec_curves(run)
        plot_class_error_confidence(run)

    plt.show()


if __name__ == '__main__':
    main()
